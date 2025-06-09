#include "thread-pool.h"
#include "moodycamel/concurrentqueue.h" // Required for ConcurrentQueue

thread_local bool ThreadPool::isWorkerThread_ = false;


ThreadPool::ThreadPool(size_t threadCount, bool complete_on_destruction)
    : threadCount_(threadCount),
      completeOnDestruction_(complete_on_destruction)
{
    // Only create worker threads and task queues if threadCount_ > 0.
    if (threadCount_ > 0)
    {
        // Initialize task queues
        taskQueues_.reset(new moodycamel::ConcurrentQueue<std::unique_ptr<ITask>>[threadCount_]);
        // queueMutexes_ is removed
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
            // No lock needed for moodycamel::ConcurrentQueue
            std::unique_ptr<ITask> discarded_task;
            while (taskQueues_[i].try_dequeue(discarded_task)) {
                // Task is removed and its std::unique_ptr is destroyed,
                // decrementing tasksCount_ is handled by the original logic
                // or by the fact that these tasks were never started.
            }
        }
        // Mark all tasks as completed (discarded) and notify any waiters
        tasksCount_.store(0, std::memory_order_relaxed);
        finishedCV_.notify_all();
    }
}

void ThreadPool::workerThread(size_t index)
{
    // Store thread ID in the vector at the index position
    // workerThreadIds_[index] = std::this_thread::get_id();

    isWorkerThread_ = true;

#if defined(__linux__)
    if (threadCount_ > 0) { // Ensure there are threads to pin
        unsigned int num_cores = std::thread::hardware_concurrency();
        if (num_cores > 0) {
            int core_id_to_pin = index % num_cores;
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(core_id_to_pin, &cpuset);
            pthread_t native_thread_handle = pthread_self();
            if (pthread_setaffinity_np(native_thread_handle, sizeof(cpu_set_t), &cpuset) != 0) {
                // Optional: std::cerr << "Failed to set affinity for thread " << index << " to core " << core_id_to_pin << std::endl;
            }
        }
    }
#elif defined(_WIN32)
    if (threadCount_ > 0) { // Ensure there are threads to pin
        unsigned int num_cores = std::thread::hardware_concurrency();
        if (num_cores > 0) {
            int core_id_to_pin = index % num_cores;
            DWORD_PTR affinityMask = 1ULL << core_id_to_pin;
            HANDLE native_thread_handle = GetCurrentThread(); // For Windows, GetCurrentThread() returns a pseudo-handle
            if (SetThreadAffinityMask(native_thread_handle, affinityMask) == 0) {
                // Optional: std::cerr << "Failed to set affinity for thread " << index << " to core " << core_id_to_pin << " (Error: " << GetLastError() << ")" << std::endl;
            }
        }
    }
#endif

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
        taskQueues_[index].try_dequeue(task);

        // 2. If none, attempt to steal a task from another thread's queue (FIFO order)
        if (!task) { // If no task from own queue, try to steal
            for (size_t offset = 1; offset < threadCount_; ++offset) {
                // It's good practice to check stopFlag_ periodically even during stealing attempts,
                // though the main check is before this block and after waking from CV.
                if (stopFlag_.load(std::memory_order_relaxed)) {
                    break; // Exit stealing loop if stopping
                }

                size_t victim_index = (index + offset) % threadCount_;

                // No lock needed for moodycamel::ConcurrentQueue's try_dequeue
                if (taskQueues_[victim_index].try_dequeue(task)) {
                    // Successfully stole a task
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

void ThreadPool::printStatus() const {
    std::vector<std::tuple<size_t, bool>> queueSnapshots;
    size_t totalPendingTasks = 0;

    if (threadCount_ > 0) { // Check if taskQueues_ was initialized
        for (size_t i = 0; i < threadCount_; ++i) {
            // No lock needed for moodycamel::ConcurrentQueue's size_approx()
            size_t queue_size = taskQueues_[i].size_approx();
            bool isActive = (queue_size > 0);
            queueSnapshots.emplace_back(queue_size, isActive);
            totalPendingTasks += queue_size;
        }
    }

    std::cout << "\n[ThreadPool] === Status ===\n";
    std::cout << "[ThreadPool] Total threads: " << threadCount_ << "\n";
    std::cout << "[ThreadPool] Accepting tasks: " << (acceptingTasks_.load(std::memory_order_acquire) ? "YES" : "NO") << "\n";
    std::cout << "[ThreadPool] Stop requested: " << (stopFlag_.load(std::memory_order_acquire) ? "YES" : "NO") << "\n";
    std::cout << "[ThreadPool] Pause requested: " << (paused_.load(std::memory_order_acquire) ? "YES" : "NO") << "\n";
    std::cout << "[ThreadPool] Total pending tasks: " << totalPendingTasks << "\n";
    std::cout << "[ThreadPool] Tasks in progress: " << (tasksCount_.load(std::memory_order_acquire) - totalPendingTasks) << "\n";

    std::cout << "[ThreadPool] Queue status:\n";
    for (size_t i = 0; i < queueSnapshots.size(); ++i) {
        std::cout << "  Queue #" << i << ": "
                  << std::get<0>(queueSnapshots[i]) << " pending tasks, "
                  << "active=" << (std::get<1>(queueSnapshots[i]) ? "YES" : "NO") << "\n";
    }

    std::cout << "[ThreadPool] Thread IDs:\n";
    for (size_t i = 0; i < threads_.size(); ++i) {
        std::cout << "  Thread #" << i << ": " << threads_[i].get_id() << "\n";
    }

    std::cout << "[ThreadPool] Current thread: " << std::this_thread::get_id()
              << (is_worker_thread() ? " (is pool worker)" : " (external)") << "\n";

    std::cout << "[ThreadPool] ========================\n";
}
