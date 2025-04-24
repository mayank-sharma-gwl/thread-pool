#pragma once
#include <deque>
#include <exception>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <atomic>
#include <stdexcept>
#include <shared_mutex>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <iostream>
#include <tuple>
#include <utility>
#include <algorithm>

class ThreadPool {
public:
    // Constructor: Launches `threadCount` worker threads.
    // If `complete_on_destruction` is true, waits for all tasks to finish on destruction.
    ThreadPool(size_t threadCount = (std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 1),
               bool   complete_on_destruction = true);
    ~ThreadPool();

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

// Reader-writer mutex to synchronize access to the shared object
#if __cplusplus >= 201703L
    mutable std::shared_mutex rwMutex; // C++17
#else
    mutable std::shared_timed_mutex rwMutex; // C++14 fallback
#endif
    
    boost::shared_mutex sharedObjRWMutex;
    
    // Enqueue a new task and get a future to its result.
    template<typename Func, typename... Args>
    auto enqueue(Func&& f, Args&&... args) -> std::future<typename std::result_of<Func(Args...)>::type> {
        using RetType  = typename std::result_of<Func(Args...)>::type;
        using TaskType = std::packaged_task<RetType()>;
        // Synchronous fallback if no worker threads are available
        if (threadCount_ == 0) {
            auto                 boundFunc = std::bind(std::forward<Func>(f), std::forward<Args>(args)...);
            TaskType             task(std::move(boundFunc));
            std::future<RetType> resultFuture = task.get_future();
            task(); // execute synchronously in the enqueuer's thread
            return resultFuture;
        }
        // Prevent enqueueing on a stopping pool
        if (!acceptingTasks_.load(std::memory_order_relaxed)) {
            throw std::runtime_error("ThreadPool is shutting down; cannot enqueue new tasks");
        }
        // Package the task and bind arguments
        auto                 boundFunc = std::bind(std::forward<Func>(f), std::forward<Args>(args)...);
        TaskType             task(std::move(boundFunc));
        std::future<RetType> resultFuture = task.get_future();
        // Wrap the packaged task into an ITask and push to a queue (round-robin distribution)
        auto                       taskWrapper = std::make_unique<Task<TaskType>>(std::move(task));
        static std::atomic<size_t> roundRobinIndex{0};
        size_t                     idx = roundRobinIndex.fetch_add(1, std::memory_order_relaxed) % threadCount_;
        {
            std::lock_guard<std::mutex> lock(queueMutexes_[idx]);
            taskQueues_[idx].push_back(std::move(taskWrapper));
        }
        // Update task count and notify one worker
        tasksCount_.fetch_add(1, std::memory_order_release);
        if (!paused_.load(std::memory_order_relaxed)) {
            tasksCV_.notify_one(); // wake a worker to handle the new task
        }
        return resultFuture;
    }

    /// Run a read-only task on sharedObject allowing parallel readers.
    /// @tparam T        Type of the shared object
    /// @tparam Callable A callable taking (T&), e.g. a lambda
    template<typename T, typename Callable>
    void enqueueThreadSafeRead(T& sharedObject, Callable&& task) {
        // Wrap the user task in a shared-lock
        auto safeRead = [this, &sharedObject, task = std::forward<Callable>(task)]() mutable {
            boost::shared_lock<boost::shared_mutex> lock(sharedObjRWMutex);
            task(sharedObject);
        };

        // Reject if pool is shutting down
        if (!acceptingTasks_.load(std::memory_order_relaxed))
            throw std::runtime_error("ThreadPool is not accepting new tasks");

        // Round-robin pick a queue
        static std::atomic<size_t> nextQueue{0};
        size_t                     queueIndex = nextQueue++ % threadCount_;

        // Enqueue the wrapped task
        {
            std::lock_guard<std::mutex> ql(queueMutexes_[queueIndex]);
            struct ReadTask : ITask {
                std::function<void()> func;
                ReadTask(std::function<void()> f) : func(std::move(f)) {}
                void run() override { func(); }
            };
            taskQueues_[queueIndex].emplace_back(std::make_unique<ReadTask>(std::move(safeRead)));
            tasksCount_.fetch_add(1, std::memory_order_release);
        }

        tasksCV_.notify_one();
    }

    /// Run a read-write task on sharedObject, exclusive of all others.
    /// @tparam T        Type of the shared object
    /// @tparam Callable A callable taking (T&), e.g. a lambda
    template<typename T, typename Callable>
    void enqueueThreadSafeWrite(T& sharedObject, Callable&& task) {
        // Wrap the user task in a unique-lock
        auto safeWrite = [this, &sharedObject, task = std::forward<Callable>(task)]() mutable {
            boost::unique_lock<boost::shared_mutex> lock(sharedObjRWMutex);
            task(sharedObject);
        };

        if (!acceptingTasks_.load(std::memory_order_relaxed))
            throw std::runtime_error("ThreadPool is not accepting new tasks");

        static std::atomic<size_t> nextQueue{0};
        size_t                     queueIndex = nextQueue++ % threadCount_;

        {
            std::lock_guard<std::mutex> ql(queueMutexes_[queueIndex]);
            struct WriteTask : ITask {
                std::function<void()> func;
                WriteTask(std::function<void()> f) : func(std::move(f)) {}
                void run() override { func(); }
            };
            taskQueues_[queueIndex].emplace_back(std::make_unique<WriteTask>(std::move(safeWrite)));
            tasksCount_.fetch_add(1, std::memory_order_release);
        }

        tasksCV_.notify_one();
    }

    template<typename Func, typename... Args>
    auto parallelRead(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        // Store arguments in a tuple to capture them in the lambda
        auto argsTuple = std::make_tuple(std::forward<Args>(args)...);
        
        // Enqueue a lambda that acquires a shared (read) lock and then executes the task
        return enqueue([this, func = std::forward<Func>(func), argsTuple = std::move(argsTuple)]() mutable {
            // Acquire shared lock for reading (multiple can hold this simultaneously)
            std::shared_lock<decltype(rwMutex)> readLock(rwMutex);
            
            // Execute the function with unpacked arguments using index sequence
            return applyTuple(std::forward<Func>(func), std::move(argsTuple));
        });
    }

    template<typename Func, typename... Args>
    auto parallelWrite(Func&& func, Args&&... args) -> std::future<decltype(func(args...))> {
        // Store arguments in a tuple to capture them in the lambda
        auto argsTuple = std::make_tuple(std::forward<Args>(args)...);
        
        // Enqueue a lambda that acquires an exclusive (write) lock and then executes the task
        return enqueue([this, func = std::forward<Func>(func), argsTuple = std::move(argsTuple)]() mutable {
            // Acquire exclusive lock for writing (blocks until no other lock is held)
            std::unique_lock<decltype(rwMutex)> writeLock(rwMutex);
            
            // Execute the function with unpacked arguments using index sequence
            return applyTuple(std::forward<Func>(func), std::move(argsTuple));
        });
    }

    // The parallelFor declaration
    template<typename IndexType, typename Func>
    void parallelFor(IndexType start,
                     IndexType end,
                     Func&& func,
                     size_t chunkSize = 0) // Use 0 to auto-compute the chunk size
    {
        if (start >= end)
            return;
        
        if (chunkSize == 0)
            chunkSize = (end - start + getThreadCount() - 1) / getThreadCount();

        std::vector<std::future<void>> futures;
        futures.reserve((end - start + chunkSize - 1) / chunkSize);
        std::exception_ptr exception = nullptr; // For exception propagation

        for (IndexType chunkStart = start; chunkStart < end; chunkStart += chunkSize) {
            IndexType chunkEnd = std::min(chunkStart + static_cast<IndexType>(chunkSize), end);

            // Perfect-forward func into task-specific copy
            auto taskFunc = std::forward<Func>(func);

            futures.emplace_back(enqueue([chunkStart, chunkEnd, func = std::move(taskFunc)]() mutable {
                for (IndexType i = chunkStart; i < chunkEnd; ++i) {
                    func(i); // Execute user function
                }
            }));
        }

        // Wait and propagate exceptions properly
        for (auto& future : futures) {
            try {
                future.get();
            } catch (...) {
                if (!exception) {
                    exception = std::current_exception();
                }
            }
        }

        if (exception) {
            std::rethrow_exception(exception);
        }
    }

    // Pause the pool: running tasks finish, new tasks will not start until resumed.
    void pause();
    // Resume a paused pool, allowing workers to continue processing tasks.
    void resume();
    // Block until all pending and active tasks have completed.
    void waitForCompletion();
    void shutdown();

    //Instrumentation
    void printStatus() const;

    size_t getThreadCount() const {
        return threadCount_;
    }
private:
    // Abstract task interface for type-erasure
    struct ITask {
        virtual ~ITask()   = default;
        virtual void run() = 0;
    };
    // Concrete task wrapper to hold and execute a callable (e.g., a packaged_task)
    template<typename Callable>
    struct Task : ITask {
        Task(Callable&& func) : func_(std::move(func)) {}
        void run() override { func_(); }

    private:
        Callable func_;
    };

    // Worker thread routine
    void workerThread(size_t index);
    // Shutdown helper: signals threads to stop and joins them.

    static thread_local bool isWorkerThread_;
    bool                     is_worker_thread() const { return isWorkerThread_; }

    size_t threadCount_;
    bool   completeOnDestruction_;

    std::vector<std::thread>                              threads_;
    std::unique_ptr<std::deque<std::unique_ptr<ITask>>[]> taskQueues_;   // per-thread task deques
    std::unique_ptr<std::mutex[]>                         queueMutexes_; // per-queue mutexes

    std::atomic<bool>   acceptingTasks_{true};
    std::atomic<bool>   paused_{false};
    std::atomic<bool>   stopFlag_{false};
    std::atomic<size_t> tasksCount_{0}; // number of tasks pending or running

    std::mutex              masterMutex_;
    std::condition_variable tasksCV_; // signals availability of tasks or stop/pause state

    std::mutex              finishedMutex_;
    std::condition_variable finishedCV_; // signals completion of all tasks

    // Helper function to unpack and apply arguments from a tuple to a function (C++14 compatible)
    template<typename Func, typename Tuple, std::size_t... Indices>
    static auto applyTupleImpl(Func&& func, Tuple&& tuple, std::index_sequence<Indices...>) 
        -> decltype(func(std::get<Indices>(std::forward<Tuple>(tuple))...)) {
        return func(std::get<Indices>(std::forward<Tuple>(tuple))...);
    }

    template<typename Func, typename Tuple>
    static auto applyTuple(Func&& func, Tuple&& tuple) 
        -> decltype(applyTupleImpl(
            std::forward<Func>(func),
            std::forward<Tuple>(tuple),
            std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{})) {
        
        return applyTupleImpl(
            std::forward<Func>(func),
            std::forward<Tuple>(tuple),
            std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{}
        );
    }
};
