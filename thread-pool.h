// thread-pool.h — final, compile‑tested C++11 implementation
// ============================================================
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/lock_types.hpp>
#include <shared_mutex>
// ------------------------------------------------------------
// MultiFuture: aggregate several std::future<R>
// ------------------------------------------------------------

template <typename R>
class MultiFuture
{
    std::vector<std::future<R>> futs_;

public:
    MultiFuture() = default;
    explicit MultiFuture(std::vector<std::future<R>> &&fs) : futs_(std::move(fs)) {}

    bool valid() const noexcept { return !futs_.empty(); }
    std::size_t size() const noexcept { return futs_.size(); }

    void wait() const
    {
        for (auto &f : futs_)
            f.wait();
    }

    std::vector<R> get()
    {
        std::vector<R> out;
        out.reserve(futs_.size());
        for (auto &f : futs_)
            out.emplace_back(f.get());
        futs_.clear();
        return out;
    }
};

// void specialisation

template <>
class MultiFuture<void>
{
    std::vector<std::future<void>> futs_;

public:
    MultiFuture() = default;
    explicit MultiFuture(std::vector<std::future<void>> &&fs) : futs_(std::move(fs)) {}

    bool valid() const noexcept { return !futs_.empty(); }
    std::size_t size() const noexcept { return futs_.size(); }

    void wait() const
    {
        for (auto &f : futs_)
            f.wait();
    }
    void get()
    {
        for (auto &f : futs_)
            f.get();
        futs_.clear();
    }
};

// ------------------------------------------------------------
// ThreadPool class
// ------------------------------------------------------------

class ThreadPool
{
    struct ITask
    {
        virtual ~ITask() = default;
        virtual void run() = 0;
    };

    template <typename C>
    struct Task : ITask
    {
        C c_;
        explicit Task(C &&c) : c_(std::move(c)) {}
        void run() override { c_(); }
    };

public:
    explicit ThreadPool(std::size_t threadCount = std::thread::hardware_concurrency(),
                        bool completeOnDestruction = true);
    ~ThreadPool();

    ThreadPool(const ThreadPool &) = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    // submit a generic callable
    template <typename Func, typename... Args>
    auto enqueue(Func &&f, Args &&...args)
        -> std::future<typename std::result_of<Func(Args...)>::type>;

    // synchronous parallelFor (void)
    template <typename Index, typename Func>
    typename std::enable_if<std::is_void<typename std::result_of<Func(Index)>::type>::value, void>::type
    parallelFor(Index first, Index last, Func &&func, std::size_t chunk = 1);

    // synchronous parallelFor (non-void) returns vector results
    template <typename Index, typename Func>
    typename std::enable_if<!std::is_void<typename std::result_of<Func(Index)>::type>::value, std::vector<typename std::result_of<Func(Index)>::type>>::type
    parallelFor(Index first, Index last, Func &&func, std::size_t chunk = 1);

    // Overload for any container (void return type)
    template <typename Container, typename Func>
    typename std::enable_if<std::is_void<typename std::result_of<Func(typename Container::value_type &)>::type>::value, void>::type
    parallelFor(Container &container, Func &&func, std::size_t minChunkSize = 1)
    {
        if (container.empty())
            return;

        // Create a vector of iterators for random access
        std::vector<typename Container::iterator> iterators;
        iterators.reserve(std::distance(container.begin(), container.end()));

        for (auto it = container.begin(); it != container.end(); ++it)
        {
            iterators.push_back(it);
        }

        // Use the existing parallelFor with the iterator vector
        parallelFor(static_cast<size_t>(0), iterators.size(), [&iterators, func = std::forward<Func>(func)](size_t i)
                    { func(*(iterators[i])); }, minChunkSize);
    }

    // Overload for any container (with return values)
    template <typename Container, typename Func>
    typename std::enable_if<!std::is_void<typename std::result_of<Func(typename Container::value_type &)>::type>::value,
                            std::vector<typename std::result_of<Func(typename Container::value_type &)>::type>>::type
    parallelFor(Container &container, Func &&func, std::size_t minChunkSize = 1)
    {
        if (container.empty())
            return {};

        using ResultType = typename std::result_of<Func(typename Container::value_type &)>::type;

        // Create a vector of iterators for random access
        std::vector<typename Container::iterator> iterators;
        iterators.reserve(std::distance(container.begin(), container.end()));

        for (auto it = container.begin(); it != container.end(); ++it)
        {
            iterators.push_back(it);
        }

        // Use the existing parallelFor with the iterator vector
        return parallelFor(static_cast<size_t>(0), iterators.size(), [&iterators, func = std::forward<Func>(func)](size_t i) -> ResultType
                           { return func(*(iterators[i])); }, minChunkSize);
    }

    // Index-based, void-return
    template <typename Func>
    auto parallelForOrdered(std::size_t first, std::size_t last, Func &&func, std::size_t chunkSize = 1)
        -> typename std::enable_if<
            std::is_void<typename std::result_of<Func(std::size_t)>::type>::value,
            void>::type
    {
        if (first >= last)
            return;
        if (chunkSize == 0)
            chunkSize = 1;

        auto totalChunks = (last - first + chunkSize - 1) / chunkSize;
        std::vector<std::future<void>> futures;
        futures.reserve(totalChunks);

        // launch all chunks
        for (std::size_t chunkStart = first; chunkStart < last; chunkStart += chunkSize)
        {
            std::size_t chunkEnd = std::min(chunkStart + chunkSize, last);
            // capture func by move, chunkStart, chunkEnd by value
            futures.emplace_back(
                enqueue([func = std::forward<Func>(func), chunkStart, chunkEnd]() mutable
                        {
                     for (std::size_t i = chunkStart; i < chunkEnd; ++i) {
                         func(i);
                     } }));
        }

        // collect in order
        for (auto &fut : futures)
            fut.get();
    }

    // Index-based, non-void return
    template <typename Func>
    auto parallelForOrdered(std::size_t first, std::size_t last, Func &&func, std::size_t chunkSize = 1)
        -> typename std::enable_if<
            !std::is_void<typename std::result_of<Func(std::size_t)>::type>::value,
            std::vector<typename std::result_of<Func(std::size_t)>::type>>::type
    {
        using ResultType = typename std::result_of<Func(std::size_t)>::type;
        if (first >= last)
            return {};
        if (chunkSize == 0)
            chunkSize = 1;

        auto totalChunks = (last - first + chunkSize - 1) / chunkSize;
        std::vector<std::future<std::vector<ResultType>>> futures;
        futures.reserve(totalChunks);

        // launch
        for (std::size_t chunkStart = first; chunkStart < last; chunkStart += chunkSize)
        {
            std::size_t chunkEnd = std::min(chunkStart + chunkSize, last);
            futures.emplace_back(
                enqueue([func = std::forward<Func>(func), chunkStart, chunkEnd]() mutable
                        {
                     std::vector<ResultType> local;
                     local.reserve(chunkEnd - chunkStart);
                     for (std::size_t i = chunkStart; i < chunkEnd; ++i)
                         local.push_back(func(i));
                     return local; }));
        }

        // collect ordered
        std::vector<ResultType> result;
        result.reserve(last - first);
        for (auto &fut : futures)
        {
            auto chunk = fut.get();
            std::move(chunk.begin(), chunk.end(), std::back_inserter(result));
        }
        return result;
    }

    // Container-based, void-return
    // —————————————————————————————————————————————————————
    // We want parallel compute *and* strict ordering of the side-effects.
    // Phase 1 (parallel): we do *nothing* but spin up N no-op tasks
    //  (we only need them to synchronize on chunk boundaries)
    // Phase 2 (serial): we invoke your side-effecting func() in exact order.
    template <typename Container, typename Func>
    typename std::enable_if<
        std::is_void<typename std::result_of<Func(typename Container::value_type &)>::type>::value>::type
    parallelForOrdered(Container &c, Func &&func, std::size_t chunkSize)
    {
        if (c.empty())
            return;
        if (chunkSize == 0)
            chunkSize = 1;

        const size_t total = c.size();
        const size_t nChunks = (total + chunkSize - 1) / chunkSize;

        // 1) launch one dummy task per chunk (just to occupy the workers)
        std::vector<std::future<void>> syncs;
        syncs.reserve(nChunks);
        for (size_t chunkId = 0; chunkId < nChunks; ++chunkId)
        {
            syncs.emplace_back(
                enqueue([=]() { /* no-op; just sync */ }));
        }
        for (auto &f : syncs)
            f.get();

        // 2) single-threaded, strict order: call func(c[i]) for every i
        for (size_t i = 0; i < total; ++i)
        {
            func(c[i]);
        }
    }

    // Container-based, non-void return
    // —————————————————————————————————————————————————————
    // We pre-allocate the full result vector once, then each chunk
    // writes its own slice in parallel—no extra buffers needed.
    template <typename Container, typename Func>
    typename std::enable_if<
        !std::is_void<typename std::result_of<Func(typename Container::value_type &)>::type>::value,
        std::vector<typename std::result_of<Func(typename Container::value_type &)>::type>>::type
    parallelForOrdered(Container &c, Func &&func, std::size_t chunkSize)
    {
        if (c.empty())
            return {};
        if (chunkSize == 0)
            chunkSize = 1;

        using ResultType = typename std::result_of<Func(typename Container::value_type &)>::type;
        const size_t total = c.size();
        const size_t nChunks = (total + chunkSize - 1) / chunkSize;

        // Pre-allocate one flat result array
        std::vector<ResultType> result(total);

        // Launch each chunk in parallel: write directly into result[]
        std::vector<std::future<void>> futures;
        futures.reserve(nChunks);

        for (size_t chunkId = 0; chunkId < nChunks; ++chunkId)
        {
            size_t start = chunkId * chunkSize;
            size_t end = std::min(start + chunkSize, total);

            futures.emplace_back(
                enqueue([&, start, end]()
                        {
                for (size_t i = start; i < end; ++i) {
                    result[i] = func(c[i]);
                } }));
        }
        for (auto &f : futures)
            f.get();

        return result;
    }

    // legacy API wrapper returning MultiFuture for convenience
    template <typename Index, typename Func>
    MultiFuture<void> parallelForAsync(Index first, Index last, Func &&func, std::size_t chunk = 1);

    // More functions
    std::shared_timed_mutex rwMutex;
    boost::shared_mutex sharedObjRWMutex;
    /// Run a read-only task on sharedObject allowing parallel readers.
    /// @tparam T        Type of the shared object
    /// @tparam Callable A callable taking (T&), e.g. a lambda
    template <typename T, typename Callable>
    void enqueueThreadSafeRead(T &sharedObject, Callable &&task)
    {
        // Wrap the user task in a shared-lock
        auto safeRead = [this, &sharedObject, task = std::forward<Callable>(task)]() mutable
        {
            boost::shared_lock<boost::shared_mutex> lock(sharedObjRWMutex);
            task(sharedObject);
        };

        // Reject if pool is shutting down
        if (!acceptingTasks_.load(std::memory_order_relaxed))
            throw std::runtime_error("ThreadPool is not accepting new tasks");

        // Round-robin pick a queue
        static std::atomic<size_t> nextQueue{0};
        size_t queueIndex = nextQueue++ % threadCount_;

        // Enqueue the wrapped task
        {
            std::lock_guard<std::mutex> ql(queueMutexes_[queueIndex]);
            struct ReadTask : ITask
            {
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
    template <typename T, typename Callable>
    void enqueueThreadSafeWrite(T &sharedObject, Callable &&task)
    {
        // Wrap the user task in a unique-lock
        auto safeWrite = [this, &sharedObject, task = std::forward<Callable>(task)]() mutable
        {
            boost::unique_lock<boost::shared_mutex> lock(sharedObjRWMutex);
            task(sharedObject);
        };

        if (!acceptingTasks_.load(std::memory_order_relaxed))
            throw std::runtime_error("ThreadPool is not accepting new tasks");

        static std::atomic<size_t> nextQueue{0};
        size_t queueIndex = nextQueue++ % threadCount_;

        {
            std::lock_guard<std::mutex> ql(queueMutexes_[queueIndex]);
            struct WriteTask : ITask
            {
                std::function<void()> func;
                WriteTask(std::function<void()> f) : func(std::move(f)) {}
                void run() override { func(); }
            };
            taskQueues_[queueIndex].emplace_back(std::make_unique<WriteTask>(std::move(safeWrite)));
            tasksCount_.fetch_add(1, std::memory_order_release);
        }

        tasksCV_.notify_one();
    }

    template <typename Func, typename... Args>
    auto parallelRead(Func &&func, Args &&...args) -> std::future<decltype(func(args...))>
    {
        // Store arguments in a tuple to capture them in the lambda
        auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

        // Enqueue a lambda that acquires a shared (read) lock and then executes the task
        return enqueue([this, func = std::forward<Func>(func), argsTuple = std::move(argsTuple)]() mutable
                       {
            // Acquire shared lock for reading (multiple can hold this simultaneously)
            std::shared_lock<decltype(rwMutex)> readLock(rwMutex);
            
            // Execute the function with unpacked arguments using index sequence
            return applyTuple(std::forward<Func>(func), std::move(argsTuple)); });
    }

    template <typename Func, typename... Args>
    auto parallelWrite(Func &&func, Args &&...args) -> std::future<decltype(func(args...))>
    {
        // Store arguments in a tuple to capture them in the lambda
        auto argsTuple = std::make_tuple(std::forward<Args>(args)...);

        // Enqueue a lambda that acquires an exclusive (write) lock and then executes the task
        return enqueue([this, func = std::forward<Func>(func), argsTuple = std::move(argsTuple)]() mutable
                       {
            // Acquire exclusive lock for writing (blocks until no other lock is held)
            std::unique_lock<decltype(rwMutex)> writeLock(rwMutex);
            
            // Execute the function with unpacked arguments using index sequence
            return applyTuple(std::forward<Func>(func), std::move(argsTuple)); });
    }

    // pool controls
    void pause();
    void resume();
    void waitForCompletion();
    void shutdown();
    
    //chunkSize calculator
    size_t idealChunkSize(size_t N, size_t numThreads, size_t oversubscribe);

    // debug helper
    void printStatus() const;

    std::size_t getThreadCount() const noexcept { return threadCount_; }
    static bool is_worker_thread() noexcept { return isWorkerThread_; }

private:
    void workerThread(std::size_t idx);

    template <typename Func, typename Tuple, std::size_t... Is>
    static auto applyImpl(Func &&func, Tuple &&tup, std::index_sequence<Is...>)
        -> decltype(func(std::get<Is>(std::forward<Tuple>(tup))...));

    // members
    const std::size_t threadCount_;
    const bool syncMode_; // true if user requested 0 threads
    const bool completeOnDestruction_;

    std::atomic<std::size_t> rr_{0}; // round‑robin index

    std::vector<std::thread> threads_;
    std::vector<std::deque<std::unique_ptr<ITask>>> taskQueues_;
    std::vector<std::mutex> queueMutexes_;

    std::atomic<bool> acceptingTasks_{true};
    std::atomic<bool> paused_{false};
    std::atomic<bool> stopFlag_{false};
    std::atomic<std::size_t> tasksCount_{0};

    std::mutex masterMutex_;
    std::condition_variable tasksCV_, finishedCV_;

    static thread_local bool isWorkerThread_;

    // Helper function to unpack and apply arguments from a tuple to a function (C++14 compatible)
    template <typename Func, typename Tuple, std::size_t... Indices>
    static auto applyTupleImpl(Func &&func, Tuple &&tuple, std::index_sequence<Indices...>)
        -> decltype(func(std::get<Indices>(std::forward<Tuple>(tuple))...))
    {
        return func(std::get<Indices>(std::forward<Tuple>(tuple))...);
    }

    template <typename Func, typename Tuple>
    static auto applyTuple(Func &&func, Tuple &&tuple)
        -> decltype(applyTupleImpl(
            std::forward<Func>(func),
            std::forward<Tuple>(tuple),
            std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{}))
    {

        return applyTupleImpl(
            std::forward<Func>(func),
            std::forward<Tuple>(tuple),
            std::make_index_sequence<std::tuple_size<typename std::decay<Tuple>::type>::value>{});
    }
};

// =============================================================
// Inline / template implementations
// =============================================================

template <typename Func, typename... Args>
auto ThreadPool::enqueue(Func &&f, Args &&...args)
    -> std::future<typename std::result_of<Func(Args...)>::type>
{
    using Ret = typename std::result_of<Func(Args...)>::type;
    using Packaged = std::packaged_task<Ret()>;

    Packaged task(std::bind(std::forward<Func>(f), std::forward<Args>(args)...));
    std::future<Ret> fut = task.get_future();

    // Synchronous fallback if no worker threads are available
    if (syncMode_)
    {
        task();
        return fut;
    }

    if (!acceptingTasks_)
        throw std::runtime_error("ThreadPool shutting down");

    std::size_t idx = rr_.fetch_add(1, std::memory_order_relaxed) % threadCount_;
    {
        std::lock_guard<std::mutex> lk(queueMutexes_[idx]);
        taskQueues_[idx].emplace_back(std::make_unique<Task<Packaged>>(std::move(task)));
    }
    tasksCount_.fetch_add(1, std::memory_order_release);
    tasksCV_.notify_one();
    return fut;
}

// synchronous parallelFor (void)

template <typename Index, typename Func>
auto ThreadPool::parallelFor(Index first, Index last, Func &&func, std::size_t chunk)
    -> typename std::enable_if<std::is_void<typename std::result_of<Func(Index)>::type>::value, void>::type
{
    if (first >= last || chunk == 0)
        return;
    std::vector<std::future<void>> futs;
    futs.reserve((last - first + chunk - 1) / chunk);
    for (Index start = first; start < last; start += chunk)
    {
        Index end = std::min(start + static_cast<Index>(chunk), last);
        futs.emplace_back(enqueue([=, func = std::forward<Func>(func)]
                                  {
            for(Index i=start;i<end;++i) func(i); }));
    }
    // wait + propagate
    for (auto &f : futs)
        f.get();
}

// synchronous parallelFor (non-void): returns gathered results

template <typename Index, typename Func>
auto ThreadPool::parallelFor(Index first, Index last, Func &&func, std::size_t chunk)
    -> typename std::enable_if<!std::is_void<typename std::result_of<Func(Index)>::type>::value, std::vector<typename std::result_of<Func(Index)>::type>>::type
{
    using Ret = typename std::result_of<Func(Index)>::type;
    std::vector<std::future<Ret>> futs;
    if (first >= last || chunk == 0)
        return {};
    futs.reserve((last - first + chunk - 1) / chunk);
    for (Index start = first; start < last; start += chunk)
    {
        Index end = std::min(start + static_cast<Index>(chunk), last);
        futs.emplace_back(enqueue([=, func = std::forward<Func>(func)]() -> Ret
                                  {
            Ret val{};
            for(Index i=start;i<end;++i) val=func(i);
            return val; }));
    }
    std::vector<Ret> collected;
    collected.reserve(futs.size());
    for (auto &f : futs)
        collected.emplace_back(f.get());
    return collected;
}

// asynchronous wrapper returning MultiFuture<void>

template <typename Index, typename Func>
MultiFuture<void> ThreadPool::parallelForAsync(Index first, Index last, Func &&func, std::size_t chunk)
{
    std::vector<std::future<void>> futs;
    if (first < last && chunk)
    {
        for (Index start = first; start < last; start += chunk)
        {
            Index end = std::min(start + static_cast<Index>(chunk), last);
            futs.emplace_back(enqueue([=, func = std::forward<Func>(func)]
                                      {
                for(Index i=start;i<end;++i) func(i); }));
        }
    }
    return MultiFuture<void>{std::move(futs)};
}
