// thread_pool.cpp — standalone implementation matching thread-pool.h
// ------------------------------------------------------------------
#include "thread-pool.h"
#include <iostream>

thread_local bool ThreadPool::isWorkerThread_ = false;

//--------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------
ThreadPool::ThreadPool(std::size_t threadCount, bool completeOnDestruction)
    : syncMode_(threadCount == 0),
      threadCount_(threadCount ? threadCount : 1),
      completeOnDestruction_(completeOnDestruction),
      taskQueues_(threadCount_),
      queueMutexes_(threadCount_)
{
    // If threadCount_ is zero, pool works synchronously (enqueue executes inline)
    if (threadCount_ > 0)
    {
        threads_.reserve(threadCount_);
        for (std::size_t i = 0; i < threadCount_; ++i)
            threads_.emplace_back(&ThreadPool::workerThread, this, i);
    }
}

//--------------------------------------------------------------------
// Destructor
//--------------------------------------------------------------------
ThreadPool::~ThreadPool()
{
    shutdown();
}

//--------------------------------------------------------------------
// Pause / Resume
//--------------------------------------------------------------------
void ThreadPool::pause()
{
    std::unique_lock<std::mutex> lock(masterMutex_);
    paused_.store(true, std::memory_order_relaxed);
    // No need to notify here; workers will check `paused_` in their wait predicate
}

void ThreadPool::resume()
{
    {
        std::unique_lock<std::mutex> lock(masterMutex_);
        if (!paused_.load(std::memory_order_relaxed))
        {
            return; // already running
        }
        paused_.store(false, std::memory_order_relaxed);
    }
    // Wake all workers so they re-evaluate the pause condition
    tasksCV_.notify_all();
}

//--------------------------------------------------------------------
// waitForCompletion (external threads only)
//--------------------------------------------------------------------
void ThreadPool::waitForCompletion()
{
    if (isWorkerThread_)
        throw std::logic_error("ThreadPool::waitForCompletion() cannot be called from worker thread");

    std::unique_lock<std::mutex> lk(masterMutex_);
    finishedCV_.wait(lk, [this]
                     { return tasksCount_.load(std::memory_order_acquire) == 0; });
}

//--------------------------------------------------------------------
// shutdown (non‑blocking, can be called multiple times)
//--------------------------------------------------------------------
void ThreadPool::shutdown()
{
    // Stop accepting new tasks
    acceptingTasks_.store(false, std::memory_order_relaxed);
    if (completeOnDestruction_)
    {
        // Wait for all ongoing tasks to finish execution
        waitForCompletion();
    }
    { // Signal all worker threads to terminate
        std::unique_lock<std::mutex> lock(masterMutex_);
        stopFlag_.store(true, std::memory_order_relaxed);
        // If paused, resume to let threads exit
        paused_.store(false, std::memory_order_relaxed);
    }
    tasksCV_.notify_all(); // wake all workers so they can exit
    // Join all threads
    for (std::thread &t : threads_)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    if (!completeOnDestruction_)
    {
        // Clear any remaining tasks without executing them
        for (size_t i = 0; i < threadCount_; ++i)
        {
            std::lock_guard<std::mutex> lock(queueMutexes_[i]);
            while (!taskQueues_[i].empty())
            {
                taskQueues_[i].pop_front(); // discard tasks
            }
        }
        // Mark all tasks as completed (discarded) and notify any waiters
        tasksCount_.store(0, std::memory_order_relaxed);
        finishedCV_.notify_all();
    }
}

//--------------------------------------------------------------------
// workerThread — work‑stealing loop
//--------------------------------------------------------------------
void ThreadPool::workerThread(std::size_t idx)
{
    isWorkerThread_ = true;
    while (true)
    {
        // Wait for work or exit signal
        {
            std::unique_lock<std::mutex> lk(masterMutex_);
            tasksCV_.wait(lk, [this]
                          { return stopFlag_.load(std::memory_order_relaxed) ||
                                   (!paused_.load(std::memory_order_relaxed) &&
                                    tasksCount_.load(std::memory_order_relaxed) > 0); });
            if (stopFlag_.load(std::memory_order_relaxed))
                return;
        }

        std::unique_ptr<ITask> task;
        // 1. Local queue (LIFO when pool has >1 thread)
        {
            std::lock_guard<std::mutex> ql(queueMutexes_[idx]);
            if (!taskQueues_[idx].empty())
            {
                if (threadCount_ == 1)
                {
                    task = std::move(taskQueues_[idx].front());
                    taskQueues_[idx].pop_front();
                }
                else
                {
                    task = std::move(taskQueues_[idx].back());
                    taskQueues_[idx].pop_back();
                }
            }
        }
        // 2. Steal from another queue (FIFO)
        if (!task && threadCount_ > 1)
        {
            for (std::size_t off = 1; off < threadCount_; ++off)
            {
                std::size_t victim = (idx + off) % threadCount_;
                std::lock_guard<std::mutex> ql(queueMutexes_[victim]);
                if (!taskQueues_[victim].empty())
                {
                    task = std::move(taskQueues_[victim].front());
                    taskQueues_[victim].pop_front();
                    break;
                }
            }
        }

        if (task)
        {
            task->run();
            if (tasksCount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                finishedCV_.notify_all();
        }
    }
}

//--------------------------------------------------------------------
// Debug helper: printStatus()
//--------------------------------------------------------------------
void ThreadPool::printStatus() const
{
    std::size_t pending = 0;
    for (const auto &q : taskQueues_)
        pending += q.size();

    std::cout << "\n[ThreadPool] Threads: " << threadCount_
              << "  Pending: " << pending
              << "  Running: " << (tasksCount_.load() - pending)
              << "  Accepting: " << (acceptingTasks_ ? "yes" : "no")
              << "  Paused: " << (paused_ ? "yes" : "no") << std::endl;
}

size_t ThreadPool::idealChunkSize(size_t N, size_t numThreads, size_t oversubscribe = 4) {
    size_t totalChunks = numThreads * oversubscribe;
    if (totalChunks == 0) totalChunks = 1;
    return (N + totalChunks - 1) / totalChunks;
}

// …later…
// size_t chunk = idealChunkSize(numElements, pool->getThreadCount(), /*F=*/4);
// pool->parallelForOrdered(0, numElements, yourFunc, chunk);
