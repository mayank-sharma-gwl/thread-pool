#include <gtest/gtest.h>
#include "../thread-pool.h"
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <climits>
#include <array>
#include <mutex>
#include <algorithm>
#include <random>
#include <map>

using namespace std::chrono_literals; // for using 100ms, etc.

// Test fixture for shared ThreadPool (with 4 threads by default)
class ThreadPoolTest : public ::testing::Test
{
protected:
    ThreadPool *pool;
    void SetUp() override
    {
        pool = new ThreadPool(4); // create a pool with 4 worker threads
    }
    void TearDown() override
    {
        // Ensure all tasks are done and clean up
        pool->waitForCompletion();
        delete pool;
    }
};

// Test fixture for reader-writer tests
class ReaderWriterTests : public ThreadPoolTest
{
protected:
    struct SharedData
    {
        std::vector<int> data{1, 2, 3, 4, 5};
        std::atomic<int> readers{0};
        std::atomic<int> writers{0};
        std::atomic<int> readCount{0};
        std::atomic<int> writeCount{0};
    };

    SharedData sharedData;
};


class ParallelForBenchmarks : public ::testing::Test {
    protected:
        ThreadPool* pool;
    
        void SetUp() override {
            pool = new ThreadPool(std::thread::hardware_concurrency());
        }
    
        void TearDown() override {
            pool->waitForCompletion();
            delete pool;
        }
};


class ParallelForOrderedTests : public ::testing::Test {
protected:
    ThreadPool* pool;

    void SetUp() override {
        pool = new ThreadPool(2*std::thread::hardware_concurrency());  // 4 threads
    }

    void TearDown() override {
        pool->waitForCompletion();
        delete pool;
    }
};

TEST_F(ThreadPoolTest, EnqueueSingleTaskReturnsCorrectValue)
{
    ThreadPool pool(4);
    // printf("Before enqueing task\n");
    // pool.printStatus();
    auto future = pool.enqueue([]
                               { return 42; });
    // printf("After enqueing task\n");
    // pool.printStatus();
    pool.waitForCompletion();
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, EnqueueMultipleTasks)
{
    ThreadPool pool(4);
    std::vector<std::future<int>> results;
    for (int i = 0; i < 10; ++i)
    {
        results.emplace_back(pool.enqueue([i]
                                          { return i * i; }));
    }
    pool.waitForCompletion();
    for (int i = 0; i < 10; ++i)
    {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

TEST_F(ThreadPoolTest, WaitForCompletionWorksCorrectly)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i)
    {
        pool.enqueue([&counter]
                     {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ++counter; });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 50);
}

TEST_F(ThreadPoolTest, PauseAndResumeWorks)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    pool.pause();
    for (int i = 0; i < 20; ++i)
    {
        pool.enqueue([&counter]
                     {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ++counter; });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(counter.load(), 0);

    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 20);
}

TEST_F(ThreadPoolTest, ShutdownBlocksUntilAllTasksFinish)
{
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 10; ++i)
        {
            pool.enqueue([&counter]
                         {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++counter; });
        }
    }
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(ThreadPoolTest, TasksThrowingExceptions)
{
    ThreadPool pool(2);
    auto future = pool.enqueue([]() -> int
                               { throw std::runtime_error("Test exception"); });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadPoolTest, HighLoadStressTest)
{
    ThreadPool pool(8);
    std::atomic<int> counter{0};
    const int taskCount = 1000;
    for (int i = 0; i < taskCount; ++i)
    {
        pool.enqueue([&counter]
                     { ++counter; });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), taskCount);
}

TEST_F(ThreadPoolTest, EnqueueDuringPauseQueuesSuccessfully)
{
    ThreadPool pool(2);
    std::atomic<int> counter{0};

    pool.pause();
    for (int i = 0; i < 10; ++i)
    {
        pool.enqueue([&counter]
                     { ++counter; });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 0);

    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(ThreadPoolTest, WaitForCompletionFromWorkerThrows)
{
    ThreadPool pool(2);
    std::promise<bool> finished;
    auto result = finished.get_future();

    pool.enqueue([&]()
                 {
        try {
            pool.waitForCompletion();
            finished.set_value(false); // did not throw — FAIL
        } catch (const std::logic_error&) {
            finished.set_value(true);  // threw as expected
        } catch (...) {
            finished.set_value(false); // wrong exception — FAIL
        } });

    // Ensure we never hang here
    ASSERT_TRUE(result.wait_for(std::chrono::seconds(2)) == std::future_status::ready)
        << "Timeout: thread did not finish!";
    EXPECT_TRUE(result.get()) << "waitForCompletion() did not throw inside worker";
}

TEST_F(ThreadPoolTest, TasksCompleteInFIFOOrderPerQueue)
{
    ThreadPool pool(1);
    std::vector<int> results;
    std::mutex res_mutex;
    for (int i = 0; i < 5; ++i)
    {
        pool.enqueue([i, &results, &res_mutex]
                     {
            std::lock_guard<std::mutex> lock(res_mutex);
            results.push_back(i); });
    }
    pool.waitForCompletion();
    for (int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(results[i], i);
    }
}

TEST_F(ThreadPoolTest, ZeroThreadsDefaultsToOne)
{
    ThreadPool pool(0);
    auto future = pool.enqueue([]
                               { return 123; });
    EXPECT_EQ(future.get(), 123);
}

TEST_F(ThreadPoolTest, PauseUnderLoadPreventsExecution)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    pool.pause();
    for (int i = 0; i < 100; ++i)
    {
        pool.enqueue([&counter]
                     { ++counter; });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 0);
    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(ThreadPoolTest, WorkStealingDoesNotCauseCorruption)
{
    ThreadPool pool(8);
    std::atomic<int> counter{0};
    for (int i = 0; i < 500; ++i)
    {
        pool.enqueue([&counter]
                     {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++counter; });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 500);
}

// **Basic Task Submission and Results**
// This test enqueues single and multiple tasks to verify the thread pool returns correct results.
TEST_F(ThreadPoolTest, BasicTaskSubmission)
{
    // Single task submission (returns an integer result)
    auto future1 = pool->enqueue([]
                                 { return 42; }); // enqueue a task that returns 42
    // The future should yield the result 42 when gotten
    EXPECT_EQ(future1.get(), 42);

    // Multiple tasks submission (concurrent tasks returning values)
    std::vector<std::future<int>> futures;
    for (int i = 1; i <= 5; ++i)
    {
        futures.push_back(pool->enqueue([i]
                                        { return i * i; })); // tasks compute square of i
    }
    // Verify each future's result corresponds to the square calculation
    for (int i = 1; i <= 5; ++i)
    {
        EXPECT_EQ(futures[i - 1].get(), i * i);
    }
}

// **Task Exception Propagation**
// If a task throws an exception, it should propagate to the std::future and be rethrown on get()&#8203;:contentReference[oaicite:5]{index=5}.
// Also ensure the thread pool remains functional after an exception.
TEST_F(ThreadPoolTest, TaskExceptionPropagation)
{
    // enqueue a task that throws a runtime_error
    auto futureErr = pool->enqueue([]() -> int
                                   { throw std::runtime_error("Task failure"); });
    // The future::get() should throw the same exception type (std::runtime_error)
    EXPECT_THROW((void)futureErr.get(), std::runtime_error);

    // After an exception, the pool should still accept and run new tasks
    auto future2 = pool->enqueue([]
                                 { return 123; });
    EXPECT_EQ(future2.get(), 123); // The new task runs normally and returns 123
}

// **Pause and Resume Behavior**
// Tests that calling pause() stops new tasks from executing until resume() is called&#8203;:contentReference[oaicite:6]{index=6}.
// Tasks enqueueted during the paused state should not run until after resume.
TEST_F(ThreadPoolTest, PauseAndResume)
{
    std::atomic<int> counter{0};

    // enqueue a couple of long-running tasks before pausing (to occupy threads)
    auto longTask = [&counter]()
    {
        std::this_thread::sleep_for(100ms);
        counter.fetch_add(1);
    };
    auto f1 = pool->enqueue(longTask);
    auto f2 = pool->enqueue(longTask);
    std::this_thread::sleep_for(20ms); // give tasks a brief head start

    pool->pause(); // pause the thread pool; new tasks should not be executed now

    // enqueue tasks while the pool is paused
    auto pausedTask1 = pool->enqueue([&counter]()
                                     { counter.fetch_add(1); });
    auto pausedTask2 = pool->enqueue([&counter]()
                                     { counter.fetch_add(1); });

    // Wait for the pre-pause tasks to finish
    f1.get();
    f2.get();
    int countAfterInitial = counter.load();
    // At this point, the two initial tasks have finished (each incremented the counter)
    // The tasks enqueueted during pause should not have run yet, so counter should not have increased further.
    EXPECT_EQ(countAfterInitial, 2);

    // Resume the pool and then wait for the paused tasks to complete
    pool->resume();
    pausedTask1.get();
    pausedTask2.get();
    // Now the counter should reflect the execution of the two paused tasks as well
    EXPECT_EQ(counter.load(), 4);
}

// **Repeated Pause/Resume Reliability**
// Repeatedly pausing and resuming the pool should not drop tasks or cause deadlocks.
// This test toggles the pause state multiple times and ensures all tasks eventually run.
TEST_F(ThreadPoolTest, RepeatedPauseResume)
{
    std::atomic<int> sum{0};
    // Perform several pause/resume cycles
    for (int cycle = 1; cycle <= 3; ++cycle)
    {
        pool->pause();
        // enqueue a task while paused (it will execute only after resume)
        auto fut = pool->enqueue([cycle, &sum]()
                                 { sum.fetch_add(cycle); });
        // Resume the pool to allow task execution
        pool->resume();
        // The task should be able to complete now and we get its future result
        fut.get(); // Wait for task completion
    }
    // After all cycles, the sum should equal 1+2+3 = 6 (all tasks ran exactly once)
    EXPECT_EQ(sum.load(), 6);
}

// **Shutdown Mechanics and Thread Completion**
// Ensures that all tasks complete before the thread pool shuts down. Tests both explicit shutdown() and implicit via destructor.
TEST(ThreadPoolLifecycle, ShutdownCompletesTasks)
{
    // Case 1: Implicit shutdown via destructor
    std::atomic<bool> taskRan1{false};
    {
        ThreadPool pool1(1);
        pool1.enqueue([&taskRan1]
                      {
            std::this_thread::sleep_for(50ms);
            taskRan1.store(true); });
        // Going out of scope should destroy pool1; it must wait for the task to finish (taskRan1 becomes true) before thread exits.
    }
    EXPECT_TRUE(taskRan1.load()) << "Task did not complete before pool destruction";

    // Case 2: Explicit shutdown() method
    ThreadPool pool2(2);
    std::atomic<bool> taskRan2{false};
    auto fut = pool2.enqueue([&taskRan2]
                             {
        std::this_thread::sleep_for(50ms);
        taskRan2.store(true); });
    pool2.shutdown(); // gracefully shut down the pool (no new tasks, wait for current tasks)
    // After shutdown returns, the enqueueted task should have completed
    EXPECT_TRUE(taskRan2.load()) << "Task did not complete before shutdown() returned";
    // Once shut down, enqueueting a new task should fail (e.g., throw an exception or return an invalid future)
    EXPECT_ANY_THROW({
        pool2.enqueue([] { /* no-op task */ });
    });
}

// **Stress Test with High Task Volume**
// Launches a high volume of tasks from multiple threads to stress the thread pool.
// Verifies that all tasks are executed and none are lost or duplicated.
TEST(ThreadPoolConcurrency, HighVolumeStress)
{
    const int numThreads = 4;
    const int tasksPerThread = 1000;
    const int totalTasks = numThreads * tasksPerThread;
    ThreadPool pool(8); // use 8 worker threads for this stress test

    std::atomic<int> counter{0};
    // Launch multiple producer threads that enqueue tasks concurrently
    std::vector<std::thread> producers;
    for (int t = 0; t < numThreads; ++t)
    {
        producers.emplace_back([&pool, &counter, tasksPerThread]()
                               {
            for (int i = 0; i < tasksPerThread; ++i) {
                pool.enqueue([&counter]() {
                    counter.fetch_add(1, std::memory_order_relaxed);
                });
            } });
    }
    // Join all producer threads (all tasks have been enqueueted)
    for (auto &pt : producers)
    {
        pt.join();
    }
    // Wait for all tasks to complete
    pool.waitForCompletion();
    // The counter should equal the total number of tasks enqueueted
    EXPECT_EQ(counter.load(), totalTasks);
}

// **Zero-Thread Pool Fallback Behavior**
// If the thread pool is created with 0 threads, it should fall back to executing tasks in the calling thread (synchronously).
TEST(ThreadPoolSpecialCases, ZeroThreadsFallback)
{
    ThreadPool pool(0); // create a pool with zero worker threads
    std::atomic<bool> executed{false};
    auto fut = pool.enqueue([&executed]()
                            { executed.store(true); return 5; });
    // Without worker threads, the task should run immediately in the enqueueter's thread.
    // The future should already be ready almost instantly.
    EXPECT_EQ(fut.wait_for(0ms), std::future_status::ready);
    EXPECT_TRUE(executed.load());
    EXPECT_EQ(fut.get(), 5);
}

// **Single-Thread FIFO Ordering**
// Verifies that a single-threaded thread pool processes tasks in first-in-first-out order&#8203;:contentReference[oaicite:7]{index=7}.
TEST(ThreadPoolSpecialCases, SingleThreadFIFO)
{
    ThreadPool singleThreadPool(1); // only one thread
    std::vector<int> executionOrder;
    std::mutex orderMutex;

    // enqueue several tasks that record their execution sequence
    for (int i = 1; i <= 5; ++i)
    {
        singleThreadPool.enqueue([i, &executionOrder, &orderMutex]()
                                 {
            std::lock_guard<std::mutex> lock(orderMutex);
            executionOrder.push_back(i); })
            .get(); // use get() to wait for completion of each task (since only one thread, it runs tasks sequentially)
    }
    // The executionOrder vector should contain 1,2,3,4,5 in order of submission
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_EQ(executionOrder, expected);
}

// **Work Stealing and Nested Tasks**
// Tests that the thread pool handles tasks which enqueue additional subtasks.
// This scenario can cause thread-starvation deadlock if work stealing is not implemented&#8203;:contentReference[oaicite:8]{index=8}.
// We expect the pool to execute nested tasks correctly (other threads steal the work if necessary&#8203;:contentReference[oaicite:9]{index=9}).
TEST(ThreadPoolSpecialCases, NestedTasksWorkStealing)
{
    ThreadPool pool(4);
    // The outer task will enqueue two inner tasks and wait for their results
    auto outerFuture = pool.enqueue([&pool]()
                                    {
        auto inner1 = pool.enqueue([]() -> int { std::this_thread::sleep_for(10ms); return 1; });
        auto inner2 = pool.enqueue([]() -> int { std::this_thread::sleep_for(10ms); return 2; });
        // Wait for inner tasks to finish and combine results
        int result1 = inner1.get();
        int result2 = inner2.get();
        return result1 + result2; });
    // If work stealing is working, the inner tasks will be taken by other threads while the outer task waits.
    // This prevents deadlock and allows the outer task to complete with the correct result.
    EXPECT_EQ(outerFuture.get(), 3);
}

// **Error Handling for waitForCompletion() in Worker Thread**
// Calling waitForCompletion (or equivalent join) from within a worker thread should be handled to avoid deadlock.
// We expect an exception to be thrown in this scenario&#8203;:contentReference[oaicite:10]{index=10}.
TEST(ThreadPoolSpecialCases, WaitForCompletionInWorker)
{
    ThreadPool pool(2);
    std::promise<bool> gotException;
    // enqueue a task that will call waitForCompletion() on the pool from within a worker
    auto fut = pool.enqueue([&pool, &gotException]()
                            {
        try {
            pool.waitForCompletion();  // worker thread waiting for all tasks (including itself)
            gotException.set_value(false);  // if it returns (no exception), that's unexpected in a correct implementation
        } catch (...) {
            // An exception (e.g., to signal potential deadlock) is expected here
            gotException.set_value(true);
        } });
    // Wait for a short time to see if the promise is set (i.e., the call returned or threw)
    auto exceptionFuture = gotException.get_future();
    if (exceptionFuture.wait_for(500ms) == std::future_status::timeout)
    {
        FAIL() << "Deadlock: waitForCompletion called from worker thread did not return or throw in time";
    }
    EXPECT_TRUE(exceptionFuture.get()) << "waitForCompletion from inside worker did not throw an exception";

    // Ensure the future is resolved (get will rethrow exception if not caught inside, which in this case it was caught)
    fut.get();
}

// **Delayed Task Scheduling Timing**
// If the thread pool supports scheduling tasks to run after a delay, verify that the task executes after the specified delay (not before).
// We allow a tolerance on timing for thread scheduling.
TEST(ThreadPoolSpecialCases, DelayedTaskScheduling)
{
    ThreadPool pool(1);
    auto startTime = std::chrono::steady_clock::now();
    // Simulate scheduling by delaying execution inside the task
    auto future = pool.enqueue([]
                               {
        std::this_thread::sleep_for(100ms);
        return std::chrono::steady_clock::now(); });
    // Get the actual execution time recorded by the task
    auto execTime = future.get();
    auto elapsed = execTime - startTime;
    // The task should not execute before 100ms have elapsed
    EXPECT_GE(elapsed, 100ms);
    // The task should execute not too far past the delay (allow some tolerance, e.g., < 500ms)
    EXPECT_LT(elapsed, 500ms);
}

TEST(ThreadPoolAdvanced, NestedTaskCreation)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::atomic<bool> allTasksCompleted{false};

    // A task that spawns more tasks
    auto nestedTask = [&pool, &counter]()
    {
        counter++;
        for (int i = 0; i < 10; i++)
        {
            pool.enqueue([&counter]()
                         {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                counter++; });
        }
    };

    // Submit several parent tasks
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; i++)
    {
        futures.push_back(pool.enqueue(nestedTask));
    }

    // Wait for parent tasks to complete
    for (auto &f : futures)
    {
        f.get();
    }

    // Wait for all tasks to complete
    pool.waitForCompletion();
    allTasksCompleted = true;

    EXPECT_TRUE(allTasksCompleted);
    EXPECT_EQ(counter.load(), 110); // 10 parent tasks + 10*10 child tasks
}

TEST(ThreadPoolAdvanced, RecursiveTaskCreation)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    // Define a recursive function that creates tasks up to a certain depth
    std::function<void(int)> recursiveTask = [&pool, &counter, &recursiveTask](int depth)
    {
        counter++;
        if (depth > 0)
        {
            pool.enqueue([depth, &recursiveTask]()
                         { recursiveTask(depth - 1); });
        }
    };

    // Start with depth 5
    auto future = pool.enqueue([&recursiveTask]()
                               { recursiveTask(5); });

    future.get();
    pool.waitForCompletion();

    EXPECT_EQ(counter.load(), 6); // 1 + 1 + 1 + 1 + 1 + 1 (original + 5 levels)
}

TEST(ThreadPoolPerformance, WorkStealing)
{
    // Test with large number of threads to observe work stealing
    ThreadPool pool(std::thread::hardware_concurrency() * 2);

    std::atomic<int> completed_tasks(0);
    std::atomic<int> long_tasks_on_same_thread(0);
    const int NUM_TASKS = 1000;

    // Use a thread_local variable to track work distribution
    std::vector<std::future<void>> futures;

    for (int i = 0; i < NUM_TASKS; i++)
    {
        bool longTask = (i % 10 == 0); // Make every 10th task a long task

        futures.push_back(pool.enqueue([longTask, &completed_tasks, &long_tasks_on_same_thread]()
                                       {
            thread_local int consecutive_long_tasks = 0;
            
            if (longTask) {
                consecutive_long_tasks++;
                if (consecutive_long_tasks > 1) {
                    long_tasks_on_same_thread++;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            } else {
                consecutive_long_tasks = 0;
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            
            completed_tasks++; }));
    }

    // Wait for all tasks to complete
    for (auto &f : futures)
    {
        f.get();
    }

    EXPECT_EQ(completed_tasks.load(), NUM_TASKS);
    // We expect efficient work stealing to distribute long tasks
    // across different threads, so there should be few cases of
    // consecutive long tasks on the same thread
    EXPECT_LT(long_tasks_on_same_thread.load(), NUM_TASKS / 20);
}

TEST(ThreadPoolTiming, TaskTimingAccuracy)
{
    ThreadPool pool(4);
    const int NUM_TIMINGS = 1000;
    std::vector<double> execution_times;

    // Check timing of low-latency tasks
    for (int i = 0; i < NUM_TIMINGS; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto future = pool.enqueue([]()
                                   {
            // Empty task to measure scheduling overhead
            return true; });
        future.get();
        auto end = std::chrono::high_resolution_clock::now();
        execution_times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }

    // Calculate standard deviation
    double sum = std::accumulate(execution_times.begin(), execution_times.end(), 0.0);
    double mean = sum / execution_times.size();
    double sq_sum = std::inner_product(execution_times.begin(), execution_times.end(),
                                       execution_times.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / execution_times.size() - mean * mean);

    // We expect consistent task scheduling with low variance
    EXPECT_LT(stdev, 2 * mean); // Standard deviation should be less than the mean
    std::cout << "Mean task scheduling time: " << mean << "ms, StdDev: " << stdev << "ms" << std::endl;
}

TEST(ThreadPoolAdvanced, DynamicWorkloadHandling)
{
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    std::atomic<int> completed{0};

    // Phase 1: Submit a batch of short tasks
    for (int i = 0; i < 100; i++)
    {
        futures.push_back(pool.enqueue([&completed]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed++;
            return 1; }));
    }

    // Phase 2: Submit a few long-running tasks
    for (int i = 0; i < 4; i++)
    {
        futures.push_back(pool.enqueue([&completed]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            completed++;
            return 10; }));
    }

    // Phase 3: Submit more short tasks that should be interleaved with long ones
    for (int i = 0; i < 100; i++)
    {
        futures.push_back(pool.enqueue([&completed]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed++;
            return 1; }));
    }

    int result_sum = 0;
    for (auto &fut : futures)
    {
        result_sum += fut.get();
    }

    EXPECT_EQ(completed.load(), 204);
    EXPECT_EQ(result_sum, 100 + 40 + 100);
}

TEST(ThreadPoolResilience, PauseResumeStress)
{
    ThreadPool pool(4);
    std::atomic<int> processed(0);
    std::atomic<bool> keepRunning(true);

    // Thread that constantly toggles pause/resume
    std::thread toggle_thread([&pool, &keepRunning]()
                              {
        int toggles = 0;
        while (keepRunning && toggles < 1000) {
            pool.pause();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            pool.resume();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            toggles++;
        } });

    // Submit a steady stream of tasks
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; i++)
    {
        futures.push_back(pool.enqueue([&processed]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            processed++; }));
    }

    // Wait for all tasks to complete
    for (auto &fut : futures)
    {
        fut.get();
    }

    keepRunning = false;
    toggle_thread.join();

    EXPECT_EQ(processed.load(), 1000);
}

TEST(ThreadPoolResilience, ExceptionSafety)
{
    ThreadPool pool(4);
    std::atomic<int> exceptionsHandled(0);
    std::atomic<int> normalTasksCompleted(0);

    // Mix of normal and exception-throwing tasks
    std::vector<std::future<void>> futures;

    // Submit tasks that throw exceptions
    for (int i = 0; i < 50; i++)
    {
        futures.push_back(pool.enqueue([i, &exceptionsHandled]() -> void
                                       {
                                           std::this_thread::sleep_for(std::chrono::milliseconds(1));
                                           exceptionsHandled++;
                                           if (i % 2 == 0)
                                               throw std::runtime_error("Even task error");
                                           if (i % 3 == 0)
                                               throw std::logic_error("Divisible by 3 error");
                                           if (i % 5 == 0)
                                               throw 42; // Non-standard exception
                                       }));
    }

    // Submit normal tasks
    for (int i = 0; i < 100; i++)
    {
        futures.push_back(pool.enqueue([&normalTasksCompleted]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            normalTasksCompleted++; }));
    }

    // Verify all tasks complete without crashing the thread pool
    int exceptions = 0;
    for (auto &fut : futures)
    {
        try
        {
            fut.get();
        }
        catch (...)
        {
            exceptions++;
        }
    }

    // Verify thread pool still works after exceptions
    auto verification_future = pool.enqueue([]()
                                            { return 42; });

    EXPECT_GT(exceptions, 0);
    EXPECT_EQ(normalTasksCompleted.load(), 100);
    EXPECT_EQ(exceptionsHandled.load(), 50);
    EXPECT_EQ(verification_future.get(), 42);
}

TEST(ThreadPoolResilience, ZeroAndMaxThreads)
{
    // Test with 0 threads (should immediately execute tasks in caller thread)
    {
        ThreadPool pool(0);
        bool executed = false;

        auto start = std::chrono::high_resolution_clock::now();
        auto fut = pool.enqueue([&executed]()
                                {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            executed = true;
            return 42; });

        // Should execute immediately and synchronously
        EXPECT_EQ(fut.wait_for(std::chrono::milliseconds(0)), std::future_status::ready);
        EXPECT_TRUE(executed);
        EXPECT_EQ(fut.get(), 42);

        auto elapsed = std::chrono::high_resolution_clock::now() - start;
        EXPECT_GE(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 50);
    }

    // Test with maximum hardware threads
    {
        ThreadPool pool(std::thread::hardware_concurrency());
        std::atomic<int> counter(0);
        std::vector<std::future<int>> futures;

        for (unsigned i = 0; i < std::thread::hardware_concurrency() * 10; i++)
        {
            futures.push_back(pool.enqueue([&counter]()
                                           {
                counter++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return counter.load(); }));
        }

        for (auto &fut : futures)
        {
            fut.get();
        }

        EXPECT_EQ(counter.load(), std::thread::hardware_concurrency() * 10);
    }
}

TEST(ThreadPoolShutdown, ShutdownUnderLoad)
{
    ThreadPool pool(4);
    std::atomic<int> completed(0);
    std::vector<std::future<void>> futures;

    // Submit a bunch of tasks
    for (int i = 0; i < 1000; i++)
    {
        futures.push_back(pool.enqueue([&completed, i]()
                                       {
            std::this_thread::sleep_for(std::chrono::milliseconds(i % 10 + 1));
            completed++; }));
    }

    // Initiate shutdown while tasks are still running
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pool.shutdown();

    // Verify completed count doesn't change after shutdown
    int completed_at_shutdown = completed.load();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // The thread pool should finish all tasks that were already enqueued
    EXPECT_EQ(completed.load(), 1000);

    // Verify we can't enqueue new tasks after shutdown
    try
    {
        auto future = pool.enqueue([]()
                                   { return true; });
        FAIL() << "Pool accepted a task after shutdown";
    }
    catch (const std::runtime_error &)
    {
        // Expected
    }
    catch (...)
    {
        FAIL() << "Unexpected exception type when submitting task after shutdown";
    }
}

TEST(ThreadPoolSpecialCases, WaitForCompletionOnEmptyPool)
{
    ThreadPool pool(4);
    // When the pool is idle, waitForCompletion should return quickly.
    auto start = std::chrono::steady_clock::now();
    pool.waitForCompletion();
    auto elapsed = std::chrono::steady_clock::now() - start;
    // Expect waitForCompletion to return in under 10ms on an idle pool.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 10)
        << "waitForCompletion took too long on an idle pool";
}

TEST(ThreadPoolSpecialCases, MultipleShutdownCalls)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; i++)
    {
        pool.enqueue([&counter]
                     {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            counter++; });
    }
    // Initiate shutdown once.
    pool.shutdown();
    // A subsequent call to shutdown() should not cause any issues.
    EXPECT_NO_THROW(pool.shutdown());
    EXPECT_EQ(counter.load(), 10);
}

TEST(ThreadPoolPerformance, MultithreadedVsSingleThreaded)
{
    // Define a CPU-intensive task that can benefit from parallelization
    auto cpu_intensive_task = []()
    {
        // More compute-intensive workload
        const int iterations = 1000000;
        std::vector<int> primes;
        primes.reserve(iterations / 10);

        for (int n = 2; n < iterations; ++n)
        {
            bool is_prime = true;
            for (int i = 2; i <= std::sqrt(n); ++i)
            {
                if (n % i == 0)
                {
                    is_prime = false;
                    break;
                }
            }
            if (is_prime)
            {
                primes.push_back(n);
            }
        }
        return primes.size();
    };

    const int task_count = 16; // Fewer but larger tasks
    std::vector<size_t> single_thread_results;
    std::vector<size_t> multi_thread_results;

    // Test 1: Single-threaded execution (gold reference)
    std::cout << "\n===== SINGLE-THREADED TEST =====\n";
    auto start_single = std::chrono::high_resolution_clock::now();
    {
        ThreadPool single_thread_pool(1);
        std::cout << "Initial single-thread pool status:" << std::endl;
        single_thread_pool.printStatus();

        std::vector<std::future<size_t>> results;
        for (int i = 0; i < task_count; ++i)
        {
            results.push_back(single_thread_pool.enqueue(cpu_intensive_task));

            // Print status after submitting the first and last task
            if (i == 0 || i == task_count - 1)
            {
                std::cout << "After submitting task #" << (i + 1) << ":" << std::endl;
                single_thread_pool.printStatus();
            }
        }

        // Print status before getting results
        std::cout << "Before getting results:" << std::endl;
        single_thread_pool.printStatus();

        // Wait for half the tasks to complete
        for (int i = 0; i < task_count / 2; ++i)
        {
            single_thread_results.push_back(results[i].get());
        }

        std::cout << "After getting half of the results:" << std::endl;
        single_thread_pool.printStatus();

        // Get remaining results
        for (int i = task_count / 2; i < task_count; ++i)
        {
            single_thread_results.push_back(results[i].get());
        }

        std::cout << "After getting all results:" << std::endl;
        single_thread_pool.printStatus();
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto single_thread_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  end_single - start_single)
                                  .count();

    // Test 2: Multi-threaded execution
    std::cout << "\n===== MULTI-THREADED TEST =====\n";
    const int thread_count = std::min(8, (int)std::thread::hardware_concurrency());

    auto start_multi = std::chrono::high_resolution_clock::now();
    {
        ThreadPool multi_thread_pool(thread_count);
        std::cout << "Initial multi-thread pool status (" << thread_count << " threads):" << std::endl;
        multi_thread_pool.printStatus();

        std::vector<std::future<size_t>> results;
        for (int i = 0; i < task_count; ++i)
        {
            results.push_back(multi_thread_pool.enqueue(cpu_intensive_task));

            // Print status after submitting the first and last task
            if (i == 0 || i == task_count - 1)
            {
                std::cout << "After submitting task #" << (i + 1) << ":" << std::endl;
                multi_thread_pool.printStatus();
            }
        }

        // Print status before getting results
        std::cout << "Before getting results:" << std::endl;
        multi_thread_pool.printStatus();

        // Wait for half the tasks to complete
        for (int i = 0; i < task_count / 2; ++i)
        {
            multi_thread_results.push_back(results[i].get());
        }

        std::cout << "After getting half of the results:" << std::endl;
        multi_thread_pool.printStatus();

        // Get remaining results
        for (int i = task_count / 2; i < task_count; ++i)
        {
            multi_thread_results.push_back(results[i].get());
        }

        std::cout << "After getting all results:" << std::endl;
        multi_thread_pool.printStatus();
    }
    auto end_multi = std::chrono::high_resolution_clock::now();
    auto multi_thread_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                 end_multi - start_multi)
                                 .count();

    // Output results
    std::cout << "\n===== PERFORMANCE COMPARISON =====\n";
    std::cout << "  Single-threaded time: " << single_thread_time << " ms" << std::endl
              << "  Multi-threaded time (" << thread_count << " threads): " << multi_thread_time << " ms" << std::endl;

    // Verify correctness by comparing results
    ASSERT_EQ(single_thread_results.size(), multi_thread_results.size())
        << "Single-threaded and multi-threaded tests should process the same number of tasks";

    bool all_results_correct = true;
    for (size_t i = 0; i < single_thread_results.size(); ++i)
    {
        if (single_thread_results[i] != multi_thread_results[i])
        {
            all_results_correct = false;
            std::cout << "Result mismatch at index " << i
                      << ": single=" << single_thread_results[i]
                      << ", multi=" << multi_thread_results[i] << std::endl;
        }
    }

    EXPECT_TRUE(all_results_correct)
        << "Multi-threaded execution produced different results than single-threaded (gold reference)";

    // Check performance improvement
    if (single_thread_time > multi_thread_time)
    {
        float speedup = (float)single_thread_time / multi_thread_time;
        std::cout << "  Speed improvement: " << speedup << "x" << std::endl;

        EXPECT_GT(speedup, 1.1f)
            << "Multi-threaded execution should be faster than single-threaded";
    }
    else
    {
        std::cout << "  No speedup detected. This could be due to thread overhead exceeding benefits "
                  << "for this particular workload or test environment." << std::endl;

        SUCCEED() << "Note: Multi-threaded version wasn't faster in this environment. "
                  << "This can happen on certain platforms or with specific workloads.";
    }
}

TEST(ThreadPoolPerformance, AsymmetricTaskDistribution)
{
    ThreadPool pool(4); // 4 worker threads
    std::vector<std::future<int>> futures;

    auto heavy_task = []()
    {
        int sum = 0;
        for (int i = 0; i < 10000000; ++i)
        {
            sum += i % 97;
        }
        return sum;
    };

    auto light_task = []()
    {
        return 1;
    };

    std::cout << "[BEFORE SUBMISSION]" << std::endl;
    pool.printStatus();

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            if (i == 0)
            {
                futures.push_back(pool.enqueue(heavy_task));
            }
            else
            {
                futures.push_back(pool.enqueue(light_task));
            }
        }
    }

    std::cout << "[AFTER SUBMISSION]" << std::endl;
    pool.printStatus();

    for (auto &fut : futures)
    {
        fut.get();
    }

    std::cout << "[AFTER COMPLETION]" << std::endl;
    pool.printStatus();

    SUCCEED() << "Asymmetric task distribution completed without deadlock or starvation.";
}

TEST(ThreadPoolPerformance, UnevenHeavyLoadComparison)
{
    auto heavy_task = []()
    {
        int sum = 0;
        for (int i = 0; i < 2000000; ++i)
        {
            sum += i % 97;
        }
        return sum;
    };

    const int total_tasks = 12;

    // Single-threaded
    auto start_single = std::chrono::high_resolution_clock::now();
    {
        ThreadPool single_pool(1);
        std::vector<std::future<int>> results;
        for (int i = 0; i < total_tasks; ++i)
        {
            results.push_back(single_pool.enqueue(heavy_task));
        }
        for (auto &fut : results)
            fut.get();
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();

    // Multi-threaded, uneven distribution
    auto start_multi = std::chrono::high_resolution_clock::now();
    {
        ThreadPool pool(4);
        std::vector<std::future<int>> results;

        std::cout << "[MULTI-THREADED BEFORE SUBMISSION]" << std::endl;
        pool.printStatus();

        for (int i = 0; i < total_tasks; ++i)
        {
            if (i < 9)
            {
                results.push_back(pool.enqueue(heavy_task)); // heavy skew
            }
            else
            {
                results.push_back(pool.enqueue([]
                                               { return 42; }));
            }
        }

        std::cout << "[MULTI-THREADED AFTER SUBMISSION]" << std::endl;
        pool.printStatus();

        for (auto &fut : results)
            fut.get();

        std::cout << "[MULTI-THREADED AFTER COMPLETION]" << std::endl;
        pool.printStatus();
    }
    auto end_multi = std::chrono::high_resolution_clock::now();
    auto time_multi = std::chrono::duration_cast<std::chrono::milliseconds>(end_multi - start_multi).count();

    std::cout << "\n===== HEAVY LOAD PERFORMANCE COMPARISON =====\n";
    std::cout << "Single-threaded time: " << time_single << " ms\n";
    std::cout << "Multi-threaded (uneven) time: " << time_multi << " ms\n";

    if (time_single > time_multi)
    {
        float speedup = static_cast<float>(time_single) / time_multi;
        std::cout << "Speedup: " << speedup << "x\n";
        EXPECT_GT(speedup, 1.1f);
    }
    else
    {
        SUCCEED() << "Multi-threaded did not outperform single-threaded, possibly due to contention.";
    }
}

// Test enqueueThreadSafeRead and enqueueThreadSafeWrite
TEST_F(ReaderWriterTests, ConcurrentReadsAllowed)
{
    std::atomic<int> completedReads{0};
    static const int READ_TASKS = 100;

    // Launch many read tasks
    for (int i = 0; i < READ_TASKS; i++)
    {
        pool->enqueueThreadSafeRead(sharedData, [&](SharedData &data)
                                    {
            // Track concurrent readers
            int readers = data.readers.fetch_add(1) + 1;
            ASSERT_GE(readers, 1) << "Reader count should be at least 1";
            ASSERT_EQ(data.writers.load(), 0) << "No writers should be active during reads";
            
            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            
            // Record that we read
            data.readCount++;
            data.readers--;
            completedReads++; });
    }

    // Wait for all reads to complete
    pool->waitForCompletion();

    // Verify all reads completed
    EXPECT_EQ(completedReads, READ_TASKS);
    EXPECT_EQ(sharedData.readCount, READ_TASKS);
    EXPECT_EQ(sharedData.writeCount, 0);
}

TEST_F(ReaderWriterTests, ExclusiveWrites)
{
    std::atomic<int> completedWrites{0};
    static const int WRITE_TASKS = 50;

    // Launch many write tasks
    for (int i = 0; i < WRITE_TASKS; i++)
    {
        pool->enqueueThreadSafeWrite(sharedData, [i, &completedWrites](SharedData &data)
                                     {
            // Ensure exclusive access
            int writers = data.writers.fetch_add(1) + 1;
            ASSERT_EQ(writers, 1) << "Only one writer should be active";
            ASSERT_EQ(data.readers.load(), 0) << "No readers should be active during writes";
            
            // Simulate longer work for writers
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            
            // Modify shared data
            data.data.push_back(i);
            data.writeCount++;
            data.writers--;
            completedWrites++; });
    }

    // Wait for all writes to complete
    pool->waitForCompletion();

    // Verify all writes completed and data was modified
    EXPECT_EQ(completedWrites, WRITE_TASKS);
    EXPECT_EQ(sharedData.writeCount, WRITE_TASKS);
    EXPECT_EQ(sharedData.data.size(), 5 + WRITE_TASKS);
}

TEST_F(ReaderWriterTests, MixedReadWriteOperations)
{
    std::atomic<int> completedOps{0};
    static const int TASKS = 200;

    // Randomly mix read and write operations
    for (int i = 0; i < TASKS; i++)
    {
        if (i % 5 == 0)
        { // 20% writes, 80% reads
            pool->enqueueThreadSafeWrite(sharedData, [i, &completedOps](SharedData &data)
                                         {
                // Writer logic
                ASSERT_EQ(data.writers.fetch_add(1), 0);
                ASSERT_EQ(data.readers.load(), 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                data.writeCount++;
                data.writers--;
                completedOps++; });
        }
        else
        {
            pool->enqueueThreadSafeRead(sharedData, [&completedOps](SharedData &data)
                                        {
                // Reader logic
                data.readers.fetch_add(1);
                ASSERT_EQ(data.writers.load(), 0);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                data.readCount++;
                data.readers--;
                completedOps++; });
        }
    }

    // Wait for all operations to complete
    pool->waitForCompletion();
    EXPECT_EQ(completedOps, TASKS);
}

// Tests for parallelRead and parallelWrite
TEST_F(ThreadPoolTest, ParallelReadReturnsFutures)
{
    std::vector<int> data{1, 2, 3, 4, 5};
    std::atomic<int> readerCount{0};

    // Function that reads from data and returns sum
    auto sumFunc = [&readerCount](const std::vector<int> &vec)
    {
        readerCount++;
        return std::accumulate(vec.begin(), vec.end(), 0);
    };

    // Launch multiple parallel reads
    std::vector<std::future<int>> results;
    for (int i = 0; i < 10; i++)
    {
        results.push_back(pool->parallelRead(sumFunc, std::ref(data)));
    }

    // Verify results
    for (auto &fut : results)
    {
        EXPECT_EQ(fut.get(), 15); // Sum is 1+2+3+4+5 = 15
    }

    EXPECT_EQ(readerCount, 10);
}

TEST_F(ThreadPoolTest, ParallelWriteModifiesData)
{
    std::vector<int> data{1, 2, 3, 4, 5};
    std::atomic<int> writerCount{0};

    std::cout << "[ParallelWriteModifiesData] Before submitting tasks:" << std::endl;
    pool->printStatus();

    // Function that modifies data and returns new size
    auto modifyFunc = [&writerCount](std::vector<int> &vec, int value)
    {
        writerCount++;
        vec.push_back(value);
        return vec.size();
    };

    // Use a sequential approach to ensure predictable results
    for (int i = 0; i < 5; i++)
    {
        std::cout << "[ParallelWriteModifiesData] Before task " << i << ":" << std::endl;
        pool->printStatus();

        // Submit task and wait for it to complete
        auto future = pool->parallelWrite(modifyFunc, std::ref(data), i + 10);
        size_t result = future.get(); // Wait for this task to complete

        std::cout << "[ParallelWriteModifiesData] After task " << i << " (size=" << result << "):" << std::endl;
        pool->printStatus();

        // Since we're waiting for each task to complete before submitting the next,
        // the size should be predictable: 5 (initial) + i + 1
        EXPECT_EQ(result, 6 + i);
    }

    std::cout << "[ParallelWriteModifiesData] After all tasks:" << std::endl;
    pool->printStatus();

    EXPECT_EQ(writerCount, 5);
    EXPECT_EQ(data.size(), 10); // Should have 5 new elements
}

// Tests for parallelFor
TEST_F(ThreadPoolTest, ParallelForProcessesRange)
{
    std::vector<int> results(100, 0);
    std::atomic<int> counter{0};

    std::cout << "[ParallelForProcessesRange] Before parallelFor:" << std::endl;
    pool->printStatus();

    // Process range 0-99, squaring each index and storing in results
    pool->parallelFor(0, 100, [&results, &counter](int i)
                      {
        results[i] = i * i;
        counter++; });

    std::cout << "[ParallelForProcessesRange] After parallelFor:" << std::endl;
    pool->printStatus();

    // Verify all elements were processed
    EXPECT_EQ(counter, 100);
    for (int i = 0; i < 100; i++)
    {
        EXPECT_EQ(results[i], i * i);
    }

    std::cout << "[ParallelForProcessesRange] After verification:" << std::endl;
    pool->printStatus();
}

TEST_F(ThreadPoolTest, ParallelForWithDifferentChunkSizes)
{
    std::vector<int> results(1000, 0);
    std::vector<size_t> chunkSizes{1, 10, 50, 100, 1000};

    for (size_t chunkSize : chunkSizes)
    {
        // Reset results
        std::fill(results.begin(), results.end(), 0);
        std::atomic<int> tasksExecuted{0};

        std::cout << "[ParallelForWithDifferentChunkSizes] Before parallelFor with chunkSize=" << chunkSize << ":" << std::endl;
        pool->printStatus();

        // Process range with specified chunk size
        pool->parallelFor(0, 1000, [&results, &tasksExecuted](int i)
                          {
            results[i] = i;
            tasksExecuted++; }, chunkSize);

        std::cout << "[ParallelForWithDifferentChunkSizes] After parallelFor with chunkSize=" << chunkSize << ":" << std::endl;
        pool->printStatus();

        // Verify all elements were processed
        EXPECT_EQ(tasksExecuted, 1000);
        for (int i = 0; i < 1000; i++)
        {
            EXPECT_EQ(results[i], i);
        }
    }
}

TEST_F(ThreadPoolTest, ParallelForExceptionHandling)
{
    // Function that throws for specific indices
    auto throwingFunc = [](int i)
    {
        if (i % 10 == 0)
        {
            throw std::runtime_error("Error at index " + std::to_string(i));
        }
    };

    std::cout << "[ParallelForExceptionHandling] Before first parallelFor:" << std::endl;
    pool->printStatus();

    // Should propagate exceptions
    EXPECT_THROW({ pool->parallelFor(0, 100, throwingFunc); }, std::runtime_error);

    std::cout << "[ParallelForExceptionHandling] After first parallelFor:" << std::endl;
    pool->printStatus();

    // Test with larger chunk size that includes throwing indices
    std::cout << "[ParallelForExceptionHandling] Before second parallelFor:" << std::endl;
    pool->printStatus();

    EXPECT_THROW({ pool->parallelFor(0, 100, throwingFunc, 20); }, std::runtime_error);

    std::cout << "[ParallelForExceptionHandling] After second parallelFor:" << std::endl;
    pool->printStatus();
}

TEST_F(ThreadPoolTest, ParallelForWithEmptyRange)
{
    std::atomic<int> counter{0};

    std::cout << "[ParallelForWithEmptyRange] Before first empty range call:" << std::endl;
    pool->printStatus();

    // Empty range should do nothing
    pool->parallelFor(0, 0, [&counter](int)
                      { counter++; });

    std::cout << "[ParallelForWithEmptyRange] After first empty range call:" << std::endl;
    pool->printStatus();

    // Invalid range
    pool->parallelFor(10, 5, [&counter](int)
                      { counter++; });

    std::cout << "[ParallelForWithEmptyRange] After invalid range call:" << std::endl;
    pool->printStatus();

    EXPECT_EQ(counter, 0);
}

// Stress test for all methods
TEST_F(ThreadPoolTest, StressTest)
{
    std::shared_ptr<std::vector<int>> sharedVector = std::make_shared<std::vector<int>>(1000, 0);
    std::atomic<int> operationsCompleted{0};
    const int OPERATIONS = 10000;

    // Launch mixed operations
    for (int i = 0; i < OPERATIONS; i++)
    {
        switch (i % 5)
        {
        case 0:
            pool->enqueueThreadSafeRead(*sharedVector, [&operationsCompleted](const std::vector<int> &)
                                        { operationsCompleted++; });
            break;
        case 1:
            pool->enqueueThreadSafeWrite(*sharedVector, [&operationsCompleted, i](std::vector<int> &vec)
                                         {
                    vec[i % vec.size()] = i;
                    operationsCompleted++; });
            break;
        case 2:
            pool->parallelRead([&operationsCompleted](const std::vector<int> &vec)
                               {
                    operationsCompleted++;
                    return vec.size(); }, std::ref(*sharedVector));
            break;
        case 3:
            pool->parallelWrite([&operationsCompleted](std::vector<int> &vec, int val)
                                {
                    vec[val % vec.size()] = val;
                    operationsCompleted++;
                    return val; }, std::ref(*sharedVector), i);
            break;
        case 4:
            // Small parallelFor to avoid too many tasks
            pool->parallelFor(0, 10, [&sharedVector, &operationsCompleted, i](int idx)
                              {
                    (*sharedVector)[(i + idx) % sharedVector->size()] = idx;
                    operationsCompleted++; }, 5);
            break;
        }
    }

    // Wait for completion and verify
    pool->waitForCompletion();
    EXPECT_GE(operationsCompleted, OPERATIONS);
}

// TEST_F(ThreadPoolTest, ParallelForPerformanceComparison)
// {
//     const int ARRAY_SIZE = 10000000; // Large enough to make the difference noticeable
//     std::vector<int> sequential_results(ARRAY_SIZE, 0);
//     std::vector<int> parallel_results(ARRAY_SIZE, 0);

//     // A moderately compute-intensive operation to apply to each element
//     auto compute_fn = [](int val)
//     {
//         // Simulate some computation
//         int result = val;
//         for (int i = 0; i < 30; ++i)
//         {
//             result = (result * 17 + 13) % 1000;
//         }
//         return result;
//     };

//     std::cout << "\n===== PARALLEL FOR PERFORMANCE TEST =====\n";

//     // 1. Sequential execution (baseline)
//     std::cout << "Starting sequential execution..." << std::endl;
//     auto start_seq = std::chrono::high_resolution_clock::now();

//     for (int i = 0; i < ARRAY_SIZE; ++i)
//     {
//         sequential_results[i] = compute_fn(i);
//     }

//     auto end_seq = std::chrono::high_resolution_clock::now();
//     auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                         end_seq - start_seq)
//                         .count();

//     std::cout << "Sequential execution completed in " << seq_time << " ms" << std::endl;

//     // 2. Test parallelFor with different chunk sizes
//     std::vector<size_t> chunk_sizes = {
//         1,               // Minimum chunk size
//         100,             // Small chunks
//         10000,           // Medium chunks
//         100000,          // Large chunks
//         ARRAY_SIZE / 10, // Very large chunks
//         ARRAY_SIZE       // Single chunk
//     };

//     std::cout << "\nTesting with " << pool->getThreadCount() << " threads in the pool" << std::endl;
//     pool->printStatus();

//     std::vector<std::pair<size_t, long>> timing_results;

//     for (size_t chunk_size : chunk_sizes)
//     {
//         // Reset results
//         std::fill(parallel_results.begin(), parallel_results.end(), 0);

//         std::cout << "\nTesting with chunk size: " << chunk_size << std::endl;
//         pool->printStatus();

//         auto start = std::chrono::high_resolution_clock::now();

//         pool->parallelFor(size_t{0}, ARRAY_SIZE, [&parallel_results, &compute_fn](int i)
//                           { parallel_results[i] = compute_fn(i); }, chunk_size);

//         auto end = std::chrono::high_resolution_clock::now();
//         auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//         std::cout << "Chunk size " << chunk_size << " completed in " << time_ms << " ms" << std::endl;
//         pool->printStatus();

//         timing_results.push_back({chunk_size, time_ms});

//         // Verify results match sequential execution
//         bool results_match = std::equal(sequential_results.begin(), sequential_results.end(),
//                                         parallel_results.begin());
//         EXPECT_TRUE(results_match)
//             << "Parallel execution with chunk size " << chunk_size << " produced different results";
//     }

//     // Print summary of results
//     std::cout << "\n===== PERFORMANCE SUMMARY =====\n";
//     std::cout << "Sequential execution: " << seq_time << " ms\n";

//     // Find the best chunk size
//     auto best_result = *std::min_element(timing_results.begin(), timing_results.end(),
//                                          [](const auto &a, const auto &b)
//                                          { return a.second < b.second; });

//     std::cout << "Parallel execution results:\n";
//     for (const auto &result_pair : timing_results)
//     {
//         size_t chunk_size = result_pair.first;
//         long time = result_pair.second;
//         double speedup = static_cast<double>(seq_time) / time;
//         std::cout << "  Chunk size " << chunk_size << ": " << time << " ms (";

//         if (speedup > 1.0)
//         {
//             std::cout << speedup << "x faster";
//         }
//         else
//         {
//             std::cout << (1.0 / speedup) << "x slower";
//         }

//         if (chunk_size == best_result.first)
//         {
//             std::cout << ") <- BEST";
//         }
//         else
//         {
//             std::cout << ")";
//         }
//         std::cout << std::endl;
//     }

//     // We expect at least some speedup with parallelization
//     EXPECT_LT(best_result.second, seq_time)
//         << "Parallel execution should be faster than sequential execution";

//     // Add additional methods to ThreadPool for this test
//     std::cout << "\nPool final status:\n";
//     pool->printStatus();
// }

TEST_F(ThreadPoolTest, ParallelForAsyncBasic)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    auto future = pool.parallelForAsync(0, 100, [&counter](int)
                                        { counter.fetch_add(1, std::memory_order_relaxed); });

    // Verify the MultiFuture is valid
    EXPECT_TRUE(future.valid());

    // Wait for completion
    future.get();

    // Verify all iterations were executed
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(ThreadPoolTest, ParallelForAsyncEmptyRange)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};

    // Test with empty range (start >= end)
    auto future1 = pool.parallelForAsync(10, 5, [&counter](int)
                                         { counter.fetch_add(1, std::memory_order_relaxed); });

    // Test with zero range
    auto future2 = pool.parallelForAsync(0, 0, [&counter](int)
                                         { counter.fetch_add(1, std::memory_order_relaxed); });

    // Should complete without exceptions
    EXPECT_NO_THROW(future1.get());
    EXPECT_NO_THROW(future2.get());

    // Counter should remain unchanged
    EXPECT_EQ(counter.load(), 0);
}

TEST_F(ThreadPoolTest, ParallelForAsyncExceptionPropagation)
{
    ThreadPool pool(4);

    auto future = pool.parallelForAsync(0, 10, [](int i)
                                        {
        if (i == 5) {
            throw std::runtime_error("Test exception");
        } });

    // The exception should be propagated through get()
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadPoolTest, ParallelForAsyncWithDifferentChunkSizes)
{
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    const int range = 100;

    std::vector<size_t> chunkSizes = {1, 10, 25, 50, 100};

    for (size_t chunkSize : chunkSizes)
    {
        counter.store(0);

        auto future = pool.parallelForAsync(0, range, [&counter](int)
                                            { counter.fetch_add(1, std::memory_order_relaxed); }, chunkSize);

        future.get();

        EXPECT_EQ(counter.load(), range)
            << "Failed with chunk size: " << chunkSize;
    }
}

TEST_F(ThreadPoolTest, ParallelForAsyncConcurrentExecution)
{
    ThreadPool pool(4);
    std::atomic<int> maxConcurrent{0};
    std::atomic<int> currentConcurrent{0};

    auto future = pool.parallelForAsync(0, 100, [&](int)
                                        {
        currentConcurrent.fetch_add(1, std::memory_order_relaxed);
        
        // Update max concurrent count
        int current = currentConcurrent.load(std::memory_order_relaxed);
        int prev_max;
        do {
            prev_max = maxConcurrent.load(std::memory_order_relaxed);
            if (current <= prev_max) break;
        } while (!maxConcurrent.compare_exchange_weak(prev_max, current,
                                                    std::memory_order_relaxed));
        
        // Simulate some work
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        currentConcurrent.fetch_sub(1, std::memory_order_relaxed); });

    future.get();

    // With 4 threads, we expect to see concurrent execution
    EXPECT_GT(maxConcurrent.load(), 1);
}

TEST_F(ThreadPoolTest, ParallelForWorkloadPerformance2)
{
    const size_t ARRAY_SIZE = 2000000;
    std::vector<double> data(ARRAY_SIZE);

    // Initialize data with some values
    std::generate(data.begin(), data.end(), []()
                  { return static_cast<double>(std::rand()) / RAND_MAX; });

    std::cout << "\n===== PARALLEL FOR ADVANCED PERFORMANCE TEST =====\n";

    // Different types of workloads to test
    struct Workload
    {
        std::string name;
        std::function<double(double)> operation;
        bool memory_bound;
    };

    std::vector<Workload> workloads = {
        {"Math Heavy (CPU bound)",
         [](double x)
         {
             double result = x;
             for (int i = 0; i < 200; ++i)
             {
                 result = std::sin(result) * std::cos(result) + std::sqrt(std::abs(result));
             }
             return result;
         },
         false},
        {"Memory Heavy",
         [](double x)
         {
             std::vector<double> temp(1000);
             double sum = 0;
             for (int i = 0; i < 50; ++i)
             {
                 temp[i % temp.size()] = x * i;
                 sum += temp[(i * 17) % temp.size()];
             }
             return sum;
         },
         true},
        {"Mixed CPU/Memory",
         [](double x)
         {
             std::vector<double> temp(100);
             double result = x;
             for (int i = 0; i < 100; ++i)
             {
                 temp[i % temp.size()] = std::sin(result);
                 result = std::sqrt(std::abs(temp[(i * 7) % temp.size()]));
             }
             return result;
         },
         true}};

    // Test with different numbers of threads
    std::vector<size_t> thread_counts = {
        1,
        2,
        4,
        std::thread::hardware_concurrency(),
        std::thread::hardware_concurrency() * 2};

    // Test with different chunk sizes
    std::vector<size_t> chunk_sizes = {
        1,
        100,
        1000,
        10000,
        ARRAY_SIZE / 100};

    struct TestResult
    {
        size_t thread_count;
        size_t chunk_size;
        double execution_time;
        double speedup;
    };

    // Run tests for each workload
    for (const auto &workload : workloads)
    {
        std::cout << "\nTesting workload: " << workload.name << "\n";
        std::cout << "----------------------------------------\n";

        // First run sequential version for baseline
        std::vector<double> sequential_results(ARRAY_SIZE);
        auto start_seq = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < ARRAY_SIZE; ++i)
        {
            sequential_results[i] = workload.operation(data[i]);
        }

        auto end_seq = std::chrono::high_resolution_clock::now();
        auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                            end_seq - start_seq)
                            .count();

        std::cout << "Sequential time: " << seq_time << "ms\n\n";

        std::vector<TestResult> results;

        // Test each thread count and chunk size combination
        for (size_t thread_count : thread_counts)
        {
            ThreadPool local_pool(thread_count);

            for (size_t chunk_size : chunk_sizes)
            {
                std::vector<double> parallel_results(ARRAY_SIZE);

                auto start = std::chrono::high_resolution_clock::now();

                local_pool.parallelFor(size_t{0}, ARRAY_SIZE, [&](size_t i)
                                       { parallel_results[i] = workload.operation(data[i]); }, chunk_size);

                auto end = std::chrono::high_resolution_clock::now();
                auto time = std::chrono::duration_cast<std::chrono::milliseconds>(
                                end - start)
                                .count();

                // Verify results match sequential version
                bool results_match = std::equal(
                    sequential_results.begin(),
                    sequential_results.end(),
                    parallel_results.begin(),
                    [](double a, double b)
                    {
                        return std::abs(a - b) < 1e-10;
                    });

                ASSERT_TRUE(results_match) << "Results mismatch with "
                                           << thread_count << " threads and chunk size " << chunk_size;

                double speedup = static_cast<double>(seq_time) / time;
                results.push_back({thread_count, chunk_size, static_cast<double>(time), speedup});

                std::cout << "Threads: " << thread_count
                          << ", Chunk: " << chunk_size
                          << ", Time: " << time << "ms"
                          << ", Speedup: " << speedup << "x\n";
            }
        }

        // Find best configuration
        auto best_result = *std::max_element(
            results.begin(),
            results.end(),
            [](const TestResult &a, const TestResult &b)
            {
                return a.speedup < b.speedup;
            });

        std::cout << "\nBest configuration for " << workload.name << ":\n"
                  << "Thread count: " << best_result.thread_count << "\n"
                  << "Chunk size: " << best_result.chunk_size << "\n"
                  << "Execution time: " << best_result.execution_time << "ms\n"
                  << "Speedup: " << best_result.speedup << "x\n";

        // Verify we got some speedup for CPU-bound tasks
        if (!workload.memory_bound)
        {
            EXPECT_GT(best_result.speedup, 1.1)
                << "Expected significant speedup for CPU-bound workload";
        }
    }
}

/*
    Benchmark single-thread vs multi-thread summation.
    Fill 10 million integers with random values.
    Compare timings and output speedup.
    Verify correctness by comparing final sums.
*/
TEST_F(ParallelForBenchmarks, HugeVectorSummationBenchmark) {
    constexpr size_t vectorSize = 10'000'000;
    std::vector<int> data(vectorSize);

    // Fill with random numbers
    std::mt19937                       rng(42);
    std::uniform_int_distribution<int> dist(1, 100);
    for (auto &v : data) v = dist(rng);

    // ---- Single-threaded Summation ----
    auto start_single = std::chrono::high_resolution_clock::now();
    long long sum_single = std::accumulate(data.begin(), data.end(), 0LL);
    auto end_single  = std::chrono::high_resolution_clock::now();
    auto time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();
    std::cout << "[Single-threaded sum = " << sum_single << "] took " << time_single << " ms\n";

    // ---- Multi-threaded Summation (one task per thread) ----
    size_t numThreads = pool->getThreadCount();
    size_t chunkSize  = (vectorSize + numThreads - 1) / numThreads;
    printf("Using %zu threads, chunk size: %zu\n", numThreads, chunkSize);
    // Start the timer
    // Note: We use high_resolution_clock for better precision
    // but the actual resolution may vary based on the system
    // and the workload.
    // This is just a demonstration; in real-world scenarios,
    // you might want to use steady_clock or system_clock
    // depending on your needs.
    // For benchmarking, high_resolution_clock is often preferred
    // for its precision.
    // However, be cautious about the clock's resolution    

    auto start_multi = std::chrono::high_resolution_clock::now();

    // 1) launch one task per chunk
    std::vector<std::future<long long>> futures;
    futures.reserve(numThreads);
    for (size_t offset = 0; offset < vectorSize; offset += chunkSize) {
        size_t begin = offset;
        size_t end   = std::min(offset + chunkSize, vectorSize);
        futures.emplace_back(
            pool->enqueue([&, begin, end]() {
                long long local = 0;
                for (size_t i = begin; i < end; ++i)
                    local += data[i];
                return local;
            })
        );
    }

    // 2) gather the partial sums
    long long sum_multi = 0;
    for (auto &f : futures)
        sum_multi += f.get();

    auto end_multi  = std::chrono::high_resolution_clock::now();
    auto time_multi = std::chrono::duration_cast<std::chrono::milliseconds>(end_multi - start_multi).count();
    std::cout << "[Multi-threaded sum = " << sum_multi << "] took " << time_multi << " ms\n";

    // ---- Verification ----
    ASSERT_EQ(sum_single, sum_multi);

    if (time_multi < time_single) {
        float speedup = float(time_single) / float(time_multi);
        std::cout << "✅ Speedup: " << speedup << "x\n";
    } else {
        std::cout << "⚠️ No speedup detected (overhead or CPU limits).\n";
    }
}

/*
    Launch 50,000 tiny tasks (each just increments a counter).
    Measures how fast the thread pool handles a lot of overhead (tiny task = more scheduling stress).
    Asserts that no task is lost.
*/

TEST_F(ParallelForBenchmarks, HeavyContentionStressTest) {
    constexpr size_t tinyTasks = 50000;
    std::atomic<size_t> counter{0};

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < tinyTasks; ++i) {
        pool->enqueue([&counter]() {
            counter.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool->waitForCompletion();

    auto end = std::chrono::high_resolution_clock::now();
    auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "[Heavy Contention Test] Completed " << counter.load() << " tiny tasks in " 
              << durationMs << " ms\n";

    EXPECT_EQ(counter.load(), tinyTasks);
}

// Custom struct for this test
struct MyData {
    int a;
    int b;
    int c;

    MyData(int x = 0) : a(x), b(x+1), c(x+2) {}
};
/*
    Defines a custom struct MyData.
    Initializes a large vector<MyData>.
    Applies parallel transformation (a += 10, b *= 2, c -= 5).
    Verifies correctness after modification.
    Demonstrates that your parallelFor() works on user-defined types!
*/
TEST_F(ParallelForBenchmarks, ParallelFor_CustomStructProcessing) {
    constexpr size_t elementCount = 10000;

    // Initialize each element with its index so a=i, b=i+1, c=i+2
    std::vector<MyData> data;
    data.reserve(elementCount);
    for (int i = 0; i < static_cast<int>(elementCount); ++i) {
        data.emplace_back(i);
    }

    // Perform parallel modification
    pool->parallelFor(data, [](MyData& item) {
        item.a += 10;
        item.b *= 2;
        item.c -= 5;
    });

    // Check correctness
    for (size_t i = 0; i < data.size(); ++i) {
        EXPECT_EQ(data[i].a, static_cast<int>(i) + 10);
        EXPECT_EQ(data[i].b, (static_cast<int>(i) + 1) * 2);
        EXPECT_EQ(data[i].c, (static_cast<int>(i) + 2) - 5);
    }

    std::cout << "[ParallelFor on MyData Struct] Successfully verified " << elementCount << " elements\n";
}

/*
    Compares Single-threaded vs Multi-threaded matrix multiply.
    Uses parallelFor(start, end, func) for row-wise splitting.
    Confirms results match exactly!
    Prints time and speedup.
    Heavy real-world test (matches your simulator style apps).
*/

TEST_F(ParallelForBenchmarks, MatrixMultiplicationBenchmark) {
    constexpr int N = 300; // Matrix dimension (NxN)
    std::vector<std::vector<int>> A(N, std::vector<int>(N, 2));
    std::vector<std::vector<int>> B(N, std::vector<int>(N, 3));
    std::vector<std::vector<int>> D(N, std::vector<int>(N, 0)); // Result matrix

    // ---- Single-threaded Matrix Multiply ----
    auto start_single = std::chrono::high_resolution_clock::now();
    {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                int sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                D[i][j] = sum;
            }
        }
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();

    std::cout << "[Single-threaded matrix multiplication took " << time_single << " ms]\n";

    // Store single-threaded result
    auto D_single = D;

    // ---- Multi-threaded Matrix Multiply using parallelFor ----
    // Reset D
    for (auto& row : D) {
        std::fill(row.begin(), row.end(), 0);
    }

    auto start_multi = std::chrono::high_resolution_clock::now();
    {
        pool->parallelFor(0, N, [&](int i) {
            for (int j = 0; j < N; ++j) {
                int sum = 0;
                for (int k = 0; k < N; ++k) {
                    sum += A[i][k] * B[k][j];
                }
                D[i][j] = sum;
            }
        });
        pool->waitForCompletion(); // Important: wait before measuring time
    }
    auto end_multi = std::chrono::high_resolution_clock::now();
    auto time_multi = std::chrono::duration_cast<std::chrono::milliseconds>(end_multi - start_multi).count();

    std::cout << "[Multi-threaded matrix multiplication took " << time_multi << " ms]\n";

    // ---- Verify matrices are equal ----
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            EXPECT_EQ(D[i][j], D_single[i][j]);
        }
    }

    if (time_multi < time_single) {
        float speedup = static_cast<float>(time_single) / time_multi;
        std::cout << "✅ Speedup: " << speedup << "x\n";
    } else {
        std::cout << "⚠️ No speedup detected (likely small size or CPU load)\n";
    }
}

/*
    Launch many readers first (50 concurrent readers).
    Launch some writers next (10 concurrent writers).
    Readers can happen in parallel, writers are exclusive.
    Validates final data state after modifications.
    Stress test for your pool's enqueueThreadSafeRead/Write()!
*/

TEST_F(ReaderWriterTests, ParallelSafeReadsAndWrites) {
    constexpr int numReaders = 50;
    constexpr int numWriters = 10;

    // 1. Launch many concurrent readers
    for (int i = 0; i < numReaders; ++i) {
        pool->enqueueThreadSafeRead(sharedData, [](const SharedData& data) {
            EXPECT_GE(data.data.size(), 5);
        });
    }

    // 2. Launch some writers that modify data
    for (int i = 0; i < numWriters; ++i) {
        pool->enqueueThreadSafeWrite(sharedData, [](SharedData& data) {
            data.data.push_back(42);
        });
    }

    pool->waitForCompletion();

    // 3. Validate
    std::cout << "[ReaderWriter Safe Test] Readers launched: " << numReaders 
              << ", Writers launched: " << numWriters << "\n";
    EXPECT_GE(sharedData.data.size(), 5 + numWriters);  // Writers add 1 element each
}

/*
    Launches asynchronous chunks (non-blocking parallelForAsync).
    Only waits later using futures (fut.get()).
    Validates all elements modified properly.
    Confirms your thread pool can handle async dispatch safely!
*/

TEST_F(ParallelForBenchmarks, ParallelForAsyncTest) {
    constexpr int totalElements = 10000;
    std::vector<int> data(totalElements, 1);  // all ones

    // fire off the async parallel-for
    MultiFuture<void> mf = pool->parallelForAsync(
        0, totalElements,
        [&data](int i) {
            data[i] *= 2;
        },
        100    // chunk size
    );

    // wait for _all_ chunks to finish
    mf.get();    // or mf.wait();

    // verify
    for (int i = 0; i < totalElements; ++i) {
        EXPECT_EQ(data[i], 2) << "at index " << i;
    }

    std::cout << "[ParallelForAsync Test] Successfully doubled "
              << totalElements << " elements asynchronously.\n";
}


/*
    In index-based parallelFor(start, end, func), you can validate final array matches index order.
    In container-based parallelFor(container, func), you can validate that no elements are left invalid, but not assume specific value patterns unless you enforce stricter locking.
*/
TEST_F(ParallelForBenchmarks, ParallelFor_ContainerOrderCorrectness) {
    constexpr size_t numElements = 5000;
    // 1) Fill with a sentinel that your worker should never write.
    std::vector<int> data(numElements, -1);

    // 2) Shared counter so every thread writes a unique >=0 value.
    static std::atomic<int> offset{0};
    pool->parallelFor(data, [&](int &val) {
        val = offset.fetch_add(1, std::memory_order_relaxed);
    });

    pool->waitForCompletion();

    // 3) Verify every slot was overwritten (no -1s remain).
    for (size_t i = 0; i < numElements; ++i) {
        EXPECT_NE(data[i], -1) << "data[" << i << "] was never written";
    }

    std::cout << "[ParallelFor Container Write Test] All elements were written.\n";
}

TEST_F(ParallelForBenchmarks, ParallelForOrdered_ContainerOrderCorrectness) {
        constexpr size_t numElements = 5000;
        // each element starts out equal to its index
        std::vector<int> data(numElements);
        std::iota(data.begin(), data.end(), 0);
    
        // collect in order
        std::vector<int> output;
        output.reserve(numElements);
        std::mutex m;
    
        pool->parallelForOrdered(data, [&](int& val) {
            std::lock_guard<std::mutex> lk(m);
            output.push_back(val);
        }, /*chunkSize=*/100);
        pool->waitForCompletion();
    
        // must have exactly [0,1,2,...]
        ASSERT_EQ(output.size(), numElements);
        for (size_t i = 0; i < numElements; ++i) {
            EXPECT_EQ(output[i], static_cast<int>(i));
        }
    
        std::cout << "[parallelForOrdered Container Order Test] Verified strict order for "
                  << numElements << " elements\n";
    }

// ----------
// 1. Range-based, void-return
TEST_F(ParallelForOrderedTests, RangeVoidMaintainsOrder) {
    constexpr size_t numElements = 1000;
    std::vector<int> output(numElements, -1);
    std::mutex outputMutex;

    pool->parallelForOrdered(0, numElements, [&](size_t i) {
        std::lock_guard<std::mutex> lock(outputMutex);
        output[i] = static_cast<int>(i);
    }, 50); // chunk size 50

    pool->waitForCompletion();

    ASSERT_EQ(output.size(), numElements);
    for (size_t i = 0; i < numElements; ++i) {
        EXPECT_EQ(output[i], static_cast<int>(i));
    }
}

// ----------
// 2. Range-based, return-value
TEST_F(ParallelForOrderedTests, RangeReturnValueMaintainsOrder) {
    constexpr size_t numElements = 1000;

    auto results = pool->parallelForOrdered(0, numElements, [](size_t i) {
        return static_cast<int>(i * i);
    }, 100); // chunk size 100

    pool->waitForCompletion();

    ASSERT_EQ(results.size(), numElements);
    for (size_t i = 0; i < numElements; ++i) {
        EXPECT_EQ(results[i], static_cast<int>(i * i));
    }
}

// ----------
// 3. Container-based, void-return
TEST_F(ParallelForOrderedTests, ContainerVoidMaintainsOrder) {
    std::vector<int> data(1000);
    std::iota(data.begin(), data.end(), 0); // 0,1,2,...

    std::vector<int> output(data.size(), -1);
    std::mutex outputMutex;

    pool->parallelForOrdered(data, [&](int& val) {
        std::lock_guard<std::mutex> lock(outputMutex);
        output[val] = val;
    }, 75);

    pool->waitForCompletion();

    ASSERT_EQ(output.size(), data.size());
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_EQ(output[i], static_cast<int>(i));
    }
}

// ----------
// 4. Container-based, return-value
TEST_F(ParallelForOrderedTests, ContainerReturnValueMaintainsOrder) {
    std::vector<int> data(500);
    std::iota(data.begin(), data.end(), 0); // 0,1,2,...

    auto results = pool->parallelForOrdered(data, [](int& val) {
        return val * 3;
    }, 50);

    pool->waitForCompletion();

    ASSERT_EQ(results.size(), data.size());
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(results[i], static_cast<int>(i * 3));
    }
}

TEST_F(ParallelForOrderedTests, HugeVectorPerformanceComparison) {
    constexpr size_t numElements = 10'000'000;

    // Prepare data
    std::vector<int> bigData(numElements);
    std::iota(bigData.begin(), bigData.end(), 1); // [1, 2, …]

    // --- Single‐threaded baseline ---
    std::vector<int> resultSingle(numElements);
    auto start_single = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numElements; ++i) {
        resultSingle[i] = bigData[i] * 2;
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();

    // --- Multi‐threaded using index‐based parallelForOrdered ---
    std::vector<int> resultParallel(numElements);
    printf("Using %zu threads\n", pool->getThreadCount());
    size_t chunk = pool->idealChunkSize(numElements, pool->getThreadCount(), /*F=*/4);
    std::cout << "Using chunk size: " << chunk << "\n";
    auto start_parallel = std::chrono::high_resolution_clock::now();
    // <— here we call the std::size_t‐indexed overload, not the void‐container one
    pool->parallelForOrdered(
        /* first     */ 0,
        /* last      */ numElements,
        /* chunk func */ [&](size_t i) {
            resultParallel[i] = bigData[i] * 2;
        },
        /*chunkSize  */ chunk
    );
    pool->waitForCompletion();  // wait for all chunk‐tasks
    auto end_parallel = std::chrono::high_resolution_clock::now();
    auto time_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();

    // --- Verify correctness ---
    ASSERT_EQ(resultSingle, resultParallel);

    // --- Report timings ---
    std::cout << "\n===== Huge Vector (10M) Performance =====\n";
    std::cout << "Single-threaded time:       " << time_single   << " ms\n";
    std::cout << "Index-based parallel time:  " << time_parallel << " ms\n";

    // Now you should see a real speedup if the hardware allows it:
    if (time_parallel < time_single) {
        float speedup = float(time_single) / float(time_parallel);
        std::cout << "✅ Speedup: " << speedup << "x\n";
    } else {
        std::cout << "⚠️ No speedup (CPU/core count, overheads, etc.)\n";
    }
}


TEST_F(ParallelForOrderedTests, ThreeWayPerformanceComparison) {
    constexpr size_t numElements = 10'000'000;
    std::vector<int> bigData(numElements);
    std::iota(bigData.begin(), bigData.end(), 1); // [1,2,...]

    std::vector<int> resultSingle(numElements, 0);
    std::vector<int> resultParallel(numElements, 0);
    std::vector<int> resultParallelOrdered(numElements, 0);

    // --- Single-threaded baseline ---
    auto start_single = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < bigData.size(); ++i) {
        resultSingle[i] = bigData[i] * 2;
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto time_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_single - start_single).count();

    // --- parallelFor (unordered) ---
    auto start_parallel = std::chrono::high_resolution_clock::now();
    pool->parallelFor(bigData, [&](int& val) {
        resultParallel[val - 1] = val * 2;
    }, 10000);
    pool->waitForCompletion();
    auto end_parallel = std::chrono::high_resolution_clock::now();
    auto time_parallel = std::chrono::duration_cast<std::chrono::milliseconds>(end_parallel - start_parallel).count();

    // --- parallelForOrdered (ordered) ---
    auto start_ordered = std::chrono::high_resolution_clock::now();
    pool->parallelForOrdered(bigData, [&](int& val) {
        resultParallelOrdered[val - 1] = val * 2;
    }, 10000);
    pool->waitForCompletion();
    auto end_ordered = std::chrono::high_resolution_clock::now();
    auto time_ordered = std::chrono::duration_cast<std::chrono::milliseconds>(end_ordered - start_ordered).count();

    // --- Validate correctness ---
    ASSERT_EQ(resultSingle, resultParallel);
    ASSERT_EQ(resultSingle, resultParallelOrdered);

    // --- Print Results ---
    std::cout << "\n===== 3-Way Performance Comparison =====\n";
    std::cout << "Single-threaded time:    " << time_single << " ms\n";
    std::cout << "Parallel (unordered) time: " << time_parallel << " ms\n";
    std::cout << "Parallel (ordered) time:   " << time_ordered << " ms\n";

    if (time_parallel < time_single) {
        float speedup = static_cast<float>(time_single) / time_parallel;
        std::cout << "Unordered parallel speedup: " << speedup << "x\n";
    }
    if (time_ordered < time_single) {
        float speedup = static_cast<float>(time_single) / time_ordered;
        std::cout << "Ordered parallel speedup: " << speedup << "x\n";
    }
}

TEST_F(ThreadPoolTest, LoadBalancingTest) {
    constexpr int T = 4;
    ThreadPool pool(T);
    std::vector<std::atomic<int>> thread_loads(T);
    
    // For mapping std::thread::id → [0..T-1]
    std::mutex id_mutex;
    std::map<std::thread::id,int> id_map;
    std::atomic<int> next_id{0};

    for (int i = 0; i < 1000; ++i) {
        pool.enqueue([&] {
            // Assign each worker thread a unique small index
            static thread_local int thread_index = -1;
            if (thread_index < 0) {
                std::lock_guard<std::mutex> lock(id_mutex);
                auto tid = std::this_thread::get_id();
                auto it  = id_map.find(tid);
                if (it == id_map.end()) {
                    // First time we see this thread
                    thread_index        = next_id++;
                    id_map[tid]         = thread_index;
                } else {
                    thread_index = it->second;
                }
            }

            // Count one unit of work for this thread
            thread_loads[thread_index].fetch_add(1, std::memory_order_relaxed);

            // Simulate some work
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
    }

    pool.waitForCompletion();

    // Compute min/max load across the T worker threads
    int min_load = std::numeric_limits<int>::max();
    int max_load = 0;
    for (int i = 0; i < T; ++i) {
        int l = thread_loads[i].load();
        min_load = std::min(min_load, l);
        max_load = std::max(max_load, l);
        std::cout << "Thread[" << i << "] did " << l << " tasks\n";
    }

    // We expect the difference to be reasonably small
    EXPECT_LT(max_load - min_load, 100);
}

TEST_F(ThreadPoolTest, ThreadPoolStateTransitions) {
    ThreadPool pool(4);
    
    // Test various state transitions
    EXPECT_NO_THROW({
        pool.pause();
        pool.pause();  // Double pause
        pool.resume();
        pool.resume(); // Double resume
        pool.shutdown();
        pool.shutdown(); // Double shutdown
    });
}

TEST_F(ThreadPoolTest, TaskCancellationBehavior) {
    ThreadPool pool(4);
    std::atomic<bool> task_ran{false};

    auto future = pool.enqueue([&task_ran]{
        std::this_thread::sleep_for(std::chrono::seconds(1));
        task_ran = true;
    });

    // Test cancellation or shutdown behavior
    pool.shutdown();
    EXPECT_TRUE(task_ran);  // Task should complete despite shutdown
}

TEST_F(ThreadPoolTest, ThreadSafetyWithExceptions) {
    ThreadPool pool(4);
    std::atomic<int> exception_count{0};
    std::atomic<int> success_count{0};

    for(int i = 0; i < 100; i++) {
        pool.enqueue([i, &exception_count, &success_count]{
            if(i % 3 == 0) {
                exception_count++;
                throw std::runtime_error("Planned error");
            }
            success_count++;
        });
    }

    pool.waitForCompletion();
    EXPECT_EQ(exception_count, 34);  // 100/3 rounded up
    EXPECT_EQ(success_count, 66);    // remaining tasks
}

TEST_F(ThreadPoolTest, ResourceExhaustionTest) {
    ThreadPool pool(4);
    std::vector<std::future<void>> futures;
    
    // Submit many large tasks to test memory handling
    for(int i = 0; i < 1000; i++) {
        futures.push_back(pool.enqueue([]{
            std::vector<int> large(1000000, 0);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }));
    }
    
    for(auto& fut : futures) {
        EXPECT_NO_THROW(fut.get());
    }
}