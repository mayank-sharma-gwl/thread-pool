// thread_pool.cpp — standalone implementation matching thread-pool.h
// ------------------------------------------------------------------
#include "thread-pool.h"
#include <iostream>
#include <thread> // for std::this_thread::yield
#include <cassert>
#include <mutex> // for std::unique_lock, std::lock_guard
#include <atomic>

thread_local bool ThreadPool::isWorkerThread_ = false;
thread_local std::size_t ThreadPool::myIndex_ = 0;

//--------------------------------------------------------------------
// Constructor
//--------------------------------------------------------------------
ThreadPool::ThreadPool(std::size_t threadCount, bool completeOnDestruction)
    : threadCount_(threadCount ? threadCount : 1),
      syncMode_(threadCount == 0),
      completeOnDestruction_(completeOnDestruction),
      queues_(threadCount_),
      stealCounts_(threadCount_)
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
    paused_.store(true, std::memory_order_release);
    // No need to notify here; workers will check `paused_` in their wait predicate
}

void ThreadPool::resume()
{
    // paused_.store(false, std::memory_order_release);
    // // release a token for each pending task so workers reawaken
    // size_t pending = 0;
    // for (size_t i = 0; i < threadCount_; ++i)
    // {
    //     std::lock_guard<std::mutex> lk(queues_[i].mutex);
    //     pending += queues_[i].tasks.size();
    // }
    // // Wake all workers so they re-evaluate the pause condition
    // tasksCV_.notify_all();
    // Hold the same masterMutex_ while unpausing & notifying
    {
        std::lock_guard<std::mutex> lock(masterMutex_);
        // Unpause with release‐semantics
        paused_.store(false, std::memory_order_release);
        // Wake all threads so they re-check the predicate
        tasksCV_.notify_all();
    }
}

//--------------------------------------------------------------------
// waitForCompletion (external threads only)
//--------------------------------------------------------------------
// void ThreadPool::waitForCompletion()
// {
//     if (isWorkerThread_)
//         throw std::logic_error("ThreadPool::waitForCompletion() cannot be called from worker thread");

//     std::unique_lock<std::mutex> lk(masterMutex_);
//     finishedCV_.wait(lk, [this]
//                      { return tasksCount_.load(std::memory_order_acquire) == 0; });
// }
void ThreadPool::waitForCompletion()
{
    if (!isWorkerThread_)
    {
        std::unique_lock<std::mutex> lk(masterMutex_);
        finishedCV_.wait(lk, [&]
                         { return tasksCount_ == 0; });
        return;
    }

    // if we *are* a worker, instead of throwing, we steal+run until done:
    while (tasksCount_.load(std::memory_order_acquire) > 1)
    {
        if (!runOneTask())
        {
            // no local work: try to steal
            stealAndRunOne();
        }
    }
}

bool ThreadPool::runOneTask()
{
    std::unique_ptr<ITask> t;
    {
        auto &Q = queues_[myIndex_];
        std::lock_guard<std::mutex> lk(Q.mutex);
        if (!Q.tasks.empty())
        {
            if (threadCount_ == 1)
            {
                t = std::move(Q.tasks.front());
                Q.tasks.pop_front();
            }
            else
            {
                t = std::move(Q.tasks.back());
                Q.tasks.pop_back();
            }
        }
    }
    if (t)
    {
        t->run();
        if (tasksCount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            finishedCV_.notify_all();
        return true;
    }
    return false;
}

void ThreadPool::stealAndRunOne()
{
    for (std::size_t off = 1; off < threadCount_; ++off)
    {
        std::size_t victim = (myIndex_ + off) % threadCount_;
        std::unique_lock<std::mutex> lk(queues_[victim].mutex, std::try_to_lock);
        if (lk.owns_lock() && !queues_[victim].tasks.empty())
        {
            auto t = std::move(queues_[victim].tasks.front());
            queues_[victim].tasks.pop_front();
            lk.unlock();
            t->run();
            if (tasksCount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
                finishedCV_.notify_all();
            return;
        }
    }
    // no work found, yield to others
    std::this_thread::yield();
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
        std::lock_guard<std::mutex> lock(masterMutex_);
        stopFlag_.store(true, std::memory_order_release);
        // If paused, resume to let threads exit
        paused_.store(false, std::memory_order_release);
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
            std::lock_guard<std::mutex> lk(queues_[i].mutex);
            queues_[i].tasks.clear();
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
    myIndex_ = idx;
    while (true)
    {
        // Wait for work or exit signal
        {
            std::unique_lock<std::mutex> lk(masterMutex_);
            tasksCV_.wait(lk, [this]
                          { return stopFlag_.load(std::memory_order_acquire) ||
                                   (!paused_.load(std::memory_order_acquire) &&
                                    tasksCount_.load(std::memory_order_acquire) > 0); });
            if (stopFlag_.load(std::memory_order_acquire))
                return;
        }

        std::unique_ptr<ITask> task;
        // 1. Local queue (LIFO when pool has >1 thread)
        {
            std::lock_guard<std::mutex> ql(queues_[idx].mutex);
            if (!queues_[idx].tasks.empty())
            {
                if (threadCount_ == 1)
                {
                    task = std::move(queues_[idx].tasks.front());
                    queues_[idx].tasks.pop_front();
                }
                else
                {
                    task = std::move(queues_[idx].tasks.back());
                    queues_[idx].tasks.pop_back();
                }
            }
        }
        // 2. Steal from another queue (FIFO)
        if (!task && threadCount_ > 1)
        {
            for (std::size_t off = 1; off < threadCount_; ++off)
            {
                std::size_t victim = (idx + off) % threadCount_;
                // std::lock_guard<std::mutex> ql(queues_[victim].mutex);
                std::unique_lock<std::mutex> ql(queues_[victim].mutex, std::try_to_lock);
                if (!ql.owns_lock() || queues_[victim].tasks.empty())
                    continue;
                if (!queues_[victim].tasks.empty())
                {
                    task = std::move(queues_[victim].tasks.front());
                    queues_[victim].tasks.pop_front();
                    // increment this worker’s steal count
                    stealCounts_[idx].fetch_add(1, std::memory_order_relaxed);
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
    // Sum up all pending tasks under each queue's lock
    for (const auto &q : queues_)
    {
        std::lock_guard<std::mutex> lk(q.mutex);
        pending += q.tasks.size();
    }

    // Load outstanding‐tasks count with acquire‐fence to pair with enqueue()'s release
    std::size_t totalOutstanding = tasksCount_.load(std::memory_order_acquire);
    // Prevent underflow if there's a momentary race
    std::size_t running = totalOutstanding > pending
                              ? totalOutstanding - pending
                              : 0;

    std::cout << "\n[ThreadPool] Threads: " << threadCount_
              << "  Pending: " << pending
              << "  Running: " << running
              << "  Accepting: " << (acceptingTasks_ ? "yes" : "no")
              << "  Paused: " << (paused_ ? "yes" : "no");
    // show per-worker steal counts
    auto steals = getStealCounts();
    std::cout << "  Steals:";
    for (size_t i = 0; i < steals.size(); ++i)
    {
        std::cout << (i == 0 ? " " : ", ") << steals[i];
    }
    std::cout << std::endl;
}

size_t ThreadPool::idealChunkSize(size_t N, size_t numThreads, size_t oversubscribe = 4)
{
    // Handle edge cases
    if (N == 0)
        return 0;
    if (numThreads == 0)
        numThreads = 1;
    if (oversubscribe == 0)
        oversubscribe = 1;

    // Calculate total desired chunks
    size_t totalChunks = numThreads * oversubscribe;

    // Avoid having more chunks than items
    totalChunks = std::min(totalChunks, N);

    // Calculate base chunk size and remainder
    size_t baseChunkSize = N / totalChunks;
    size_t remainder = N % totalChunks;

    // If chunk size is too small, reduce number of chunks
    const size_t MIN_CHUNK_SIZE = 16; // Minimum items per chunk to amortize overhead
    if (baseChunkSize < MIN_CHUNK_SIZE)
    {
        baseChunkSize = std::min(MIN_CHUNK_SIZE, N);
        totalChunks = N / baseChunkSize + (remainder != 0 ? 1 : 0);
        return baseChunkSize;
    }

    // Round up chunk size if there's remainder to evenly distribute work
    return baseChunkSize + (remainder != 0 ? 1 : 0);
}

// …later…
// size_t chunk = idealChunkSize(numElements, pool->getThreadCount(), /*F=*/4);
// pool->parallelForOrdered(0, numElements, yourFunc, chunk);
// ----------------------------------------------------------------------------
// Accessors for testing / introspection
// ----------------------------------------------------------------------------

std::vector<size_t> ThreadPool::getStealCounts() const
{
    std::vector<size_t> out(threadCount_);
    for (size_t i = 0; i < threadCount_; ++i)
    {
        out[i] = stealCounts_[i].load(std::memory_order_relaxed);
    }
    return out;
}

std::vector<size_t> ThreadPool::getQueueSizes() const
{
    std::vector<size_t> out(threadCount_);
    for (size_t i = 0; i < threadCount_; ++i)
    {
        std::lock_guard<std::mutex> lk(queues_[i].mutex);
        out[i] = queues_[i].tasks.size();
    }
    return out;
}