#pragma once
#include <deque>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <stdexcept>

class ThreadPool
{
public:
    // Constructor: Launches `threadCount` worker threads.
    // If `complete_on_destruction` is true, waits for all tasks to finish on destruction.
    ThreadPool(size_t threadCount = (std::thread::hardware_concurrency() > 0
                                         ? std::thread::hardware_concurrency()
                                         : 1),
               bool complete_on_destruction = true);
    ~ThreadPool();

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    // Enqueue a new task and get a future to its result.
    template <typename Func, typename... Args>
    auto enqueue(Func &&f, Args &&...args)
        -> std::future<typename std::result_of<Func(Args...)>::type>
    {
        using RetType = typename std::result_of<Func(Args...)>::type;
        using TaskType = std::packaged_task<RetType()>;
        // Synchronous fallback if no worker threads are available
        if (threadCount_ == 0)
        {
            auto boundFunc = std::bind(std::forward<Func>(f), std::forward<Args>(args)...);
            TaskType task(std::move(boundFunc));
            std::future<RetType> resultFuture = task.get_future();
            task(); // execute synchronously in the enqueuer's thread
            return resultFuture;
        }
        // Prevent enqueueing on a stopping pool
        if (!acceptingTasks_.load(std::memory_order_relaxed))
        {
            throw std::runtime_error("ThreadPool is shutting down; cannot enqueue new tasks");
        }
        // Package the task and bind arguments
        auto boundFunc = std::bind(std::forward<Func>(f), std::forward<Args>(args)...);
        TaskType task(std::move(boundFunc));
        std::future<RetType> resultFuture = task.get_future();
        // Wrap the packaged task into an ITask and push to a queue (round-robin distribution)
        auto taskWrapper = std::make_unique<Task<TaskType>>(std::move(task));
        static std::atomic<size_t> roundRobinIndex{0};
        size_t idx = roundRobinIndex.fetch_add(1, std::memory_order_relaxed) % threadCount_;
        {
            std::lock_guard<std::mutex> lock(queueMutexes_[idx]);
            taskQueues_[idx].push_back(std::move(taskWrapper));
        }
        // Update task count and notify one worker
        tasksCount_.fetch_add(1, std::memory_order_release);
        if (!paused_.load(std::memory_order_relaxed))
        {
            tasksCV_.notify_one(); // wake a worker to handle the new task
        }
        return resultFuture;
    }

    /// Submit a task that operates on a shared object with internal synchronization.
    /// @tparam T        Type of the shared object
    /// @tparam Callable A callable taking (T&), e.g. a lambda
    template <typename T, typename Callable>
    void executeOnShared(T& sharedObject, Callable&& task);

    // Pause the pool: running tasks finish, new tasks will not start until resumed.
    void pause();
    // Resume a paused pool, allowing workers to continue processing tasks.
    void resume();
    // Block until all pending and active tasks have completed.
    void waitForCompletion();
    void shutdown();

private:
    // Abstract task interface for type-erasure
    struct ITask
    {
        virtual ~ITask() = default;
        virtual void run() = 0;
    };
    // Concrete task wrapper to hold and execute a callable (e.g., a packaged_task)
    template <typename Callable>
    struct Task : ITask
    {
        Task(Callable &&func) : func_(std::move(func)) {}
        void run() override { func_(); }

    private:
        Callable func_;
    };

    // Worker thread routine
    void workerThread(size_t index);
    // Shutdown helper: signals threads to stop and joins them.

    static thread_local bool isWorkerThread_;
    bool is_worker_thread() const
    {
        return isWorkerThread_;
    }

    size_t threadCount_;
    bool completeOnDestruction_;

    std::vector<std::thread> threads_;
    std::unique_ptr<std::deque<std::unique_ptr<ITask>>[]> taskQueues_; // per-thread task deques
    std::unique_ptr<std::mutex[]> queueMutexes_;                       // per-queue mutexes

    std::atomic<bool> acceptingTasks_{true};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stopFlag_{false};
    std::atomic<size_t> tasksCount_{0}; // number of tasks pending or running

    std::mutex masterMutex_;
    std::condition_variable tasksCV_; // signals availability of tasks or stop/pause state

    std::mutex finishedMutex_;
    std::condition_variable finishedCV_; // signals completion of all tasks
};
