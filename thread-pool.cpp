#include "thread-pool.h"

thread_local bool ThreadPool::isWorkerThread_ = false;

ThreadPool::ThreadPool(size_t threadCount, bool complete_on_destruction)
    : threadCount_(threadCount),
      completeOnDestruction_(complete_on_destruction)
{
    // Only create worker threads and task queues if threadCount_ > 0.
    if (threadCount_ > 0)
    {
        // Initialize task queues and corresponding mutexes
        taskQueues_.reset(new std::deque<std::unique_ptr<ITask>>[threadCount_]);
        queueMutexes_.reset(new std::mutex[threadCount_]);
        // Launch worker threads
        threads_.reserve(threadCount_);
        for (size_t i = 0; i < threadCount_; ++i)
        {
            threads_.emplace_back(&ThreadPool::workerThread, this, i);
        }
    }
}

ThreadPool::~ThreadPool()
{
    shutdown();
}

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

void ThreadPool::waitForCompletion()
{
    if (is_worker_thread())
    {
        throw std::logic_error("waitForCompletion() called from worker thread");
    }
    std::unique_lock<std::mutex> lock(finishedMutex_);
    finishedCV_.wait(lock, [this]()
                     { return tasksCount_.load(std::memory_order_acquire) == 0; });
    // Returns when no tasks are pending or running
}

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

void ThreadPool::workerThread(size_t index)
{
    isWorkerThread_ = true;
    for (;;)
    {
        // Wait for a task to be available or for shutdown/pause signals
        std::unique_lock<std::mutex> lock(masterMutex_);
        tasksCV_.wait(lock, [this]()
                      { return stopFlag_.load(std::memory_order_relaxed) ||
                               (!paused_.load(std::memory_order_relaxed) &&
                                tasksCount_.load(std::memory_order_relaxed) > 0); });
        if (stopFlag_.load(std::memory_order_relaxed))
        {
            // Shutdown signal received
            return;
        }
        lock.unlock(); // release master lock before accessing task queues

        // Fetch a task if available
        std::unique_ptr<ITask> task;
        // 1. Try to get a task from this thread's own queue
        {
            std::lock_guard<std::mutex> qlock(queueMutexes_[index]);
            if (!taskQueues_[index].empty())
            {
                // If only one thread, use FIFO (pop_front); otherwise use LIFO (pop_back)
                if (threadCount_ == 1)
                {
                    task = std::move(taskQueues_[index].front());
                    taskQueues_[index].pop_front();
                }
                else
                {
                    task = std::move(taskQueues_[index].back());
                    taskQueues_[index].pop_back();
                }
            }
        }
        // 2. If none, attempt to steal a task from another thread's queue (FIFO order)
        if (!task)
        {
            for (size_t offset = 1; offset < threadCount_; ++offset)
            {
                size_t victim = (index + offset) % threadCount_;
                std::lock_guard<std::mutex> qlock(queueMutexes_[victim]);
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
            // Execute the retrieved task outside of any locks
            task->run();
            // Decrement the task counter; if this was the last task, notify waiters
            if (tasksCount_.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                finishedCV_.notify_all();
            }
            // Continue without delay
            continue;
        }
        // If no task was found, loop back to waiting.
    }
}
