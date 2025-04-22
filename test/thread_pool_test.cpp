#include <gtest/gtest.h>
#include "../thread-pool.h"
#include <vector>
#include <atomic>
#include <chrono>
#include <thread>
#include <future>
#include <stdexcept>
#include <numeric> 

using namespace std::chrono_literals;  // for using 100ms, etc.

// Test fixture for shared ThreadPool (with 4 threads by default)
class ThreadPoolTest : public ::testing::Test {
protected:
    ThreadPool* pool;
    void SetUp() override {
        pool = new ThreadPool(4);  // create a pool with 4 worker threads
    }
    void TearDown() override {
        // Ensure all tasks are done and clean up
        pool->waitForCompletion();
        delete pool;
    }
};

TEST_F(ThreadPoolTest, EnqueueSingleTaskReturnsCorrectValue) {
    ThreadPool pool(4);
    auto future = pool.enqueue([] { return 42; });
    pool.waitForCompletion();
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadPoolTest, EnqueueMultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> results;
    for (int i = 0; i < 10; ++i) {
        results.emplace_back(pool.enqueue([i] { return i * i; }));
    }
    pool.waitForCompletion();
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(results[i].get(), i * i);
    }
}

TEST_F(ThreadPoolTest, WaitForCompletionWorksCorrectly) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 50; ++i) {
        pool.enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            ++counter;
        });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 50);
}

TEST_F(ThreadPoolTest, PauseAndResumeWorks) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    pool.pause();
    for (int i = 0; i < 20; ++i) {
        pool.enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            ++counter;
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(counter.load(), 0);

    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 20);
}

TEST_F(ThreadPoolTest, ShutdownBlocksUntilAllTasksFinish) {
    std::atomic<int> counter{0};
    {
        ThreadPool pool(4);
        for (int i = 0; i < 10; ++i) {
            pool.enqueue([&counter] {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                ++counter;
            });
        }
    }
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(ThreadPoolTest, TasksThrowingExceptions) {
    ThreadPool pool(2);
    auto future = pool.enqueue([]() -> int {
        throw std::runtime_error("Test exception");
    });
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadPoolTest, HighLoadStressTest) {
    ThreadPool pool(8);
    std::atomic<int> counter{0};
    const int taskCount = 1000;
    for (int i = 0; i < taskCount; ++i) {
        pool.enqueue([&counter] { ++counter; });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), taskCount);
}

TEST_F(ThreadPoolTest, EnqueueDuringPauseQueuesSuccessfully) {
    ThreadPool pool(2);
    std::atomic<int> counter{0};

    pool.pause();
    for (int i = 0; i < 10; ++i) {
        pool.enqueue([&counter] {
            ++counter;
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 0);

    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(ThreadPoolTest, WaitForCompletionFromWorkerThrows) {
    ThreadPool pool(2);
    std::promise<bool> finished;
    auto result = finished.get_future();

    pool.enqueue([&]() {
        try {
            pool.waitForCompletion();
            finished.set_value(false); // did not throw — FAIL
        } catch (const std::logic_error&) {
            finished.set_value(true);  // threw as expected
        } catch (...) {
            finished.set_value(false); // wrong exception — FAIL
        }
    });

    // Ensure we never hang here
    ASSERT_TRUE(result.wait_for(std::chrono::seconds(2)) == std::future_status::ready)
        << "Timeout: thread did not finish!";
    EXPECT_TRUE(result.get()) << "waitForCompletion() did not throw inside worker";
}


TEST_F(ThreadPoolTest, TasksCompleteInFIFOOrderPerQueue) {
    ThreadPool pool(1);
    std::vector<int> results;
    std::mutex res_mutex;
    for (int i = 0; i < 5; ++i) {
        pool.enqueue([i, &results, &res_mutex] {
            std::lock_guard<std::mutex> lock(res_mutex);
            results.push_back(i);
        });
    }
    pool.waitForCompletion();
    for (int i = 0; i < 5; ++i) {
        EXPECT_EQ(results[i], i);
    }
}

TEST_F(ThreadPoolTest, ZeroThreadsDefaultsToOne) {
    ThreadPool pool(0);
    auto future = pool.enqueue([] { return 123; });
    EXPECT_EQ(future.get(), 123);
}

TEST_F(ThreadPoolTest, PauseUnderLoadPreventsExecution) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    pool.pause();
    for (int i = 0; i < 100; ++i) {
        pool.enqueue([&counter] {
            ++counter;
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 0);
    pool.resume();
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 100);
}

TEST_F(ThreadPoolTest, WorkStealingDoesNotCauseCorruption) {
    ThreadPool pool(8);
    std::atomic<int> counter{0};
    for (int i = 0; i < 500; ++i) {
        pool.enqueue([&counter] {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++counter;
        });
    }
    pool.waitForCompletion();
    EXPECT_EQ(counter.load(), 500);
}

// **Basic Task Submission and Results**  
// This test enqueues single and multiple tasks to verify the thread pool returns correct results.
TEST_F(ThreadPoolTest, BasicTaskSubmission) { 
    // Single task submission (returns an integer result)
    auto future1 = pool->enqueue([] { return 42; });  // enqueue a task that returns 42
    // The future should yield the result 42 when gotten
    EXPECT_EQ(future1.get(), 42);

    // Multiple tasks submission (concurrent tasks returning values)
    std::vector<std::future<int>> futures;
    for (int i = 1; i <= 5; ++i) {
        futures.push_back(pool->enqueue([i] { return i * i; }));  // tasks compute square of i
    }
    // Verify each future's result corresponds to the square calculation
    for (int i = 1; i <= 5; ++i) {
        EXPECT_EQ(futures[i-1].get(), i * i);
    }
}

// **Task Exception Propagation**  
// If a task throws an exception, it should propagate to the std::future and be rethrown on get()&#8203;:contentReference[oaicite:5]{index=5}. 
// Also ensure the thread pool remains functional after an exception.
TEST_F(ThreadPoolTest, TaskExceptionPropagation) {
    // enqueue a task that throws a runtime_error
    auto futureErr = pool->enqueue([]() -> int {
        throw std::runtime_error("Task failure");
    });
    // The future::get() should throw the same exception type (std::runtime_error)
    EXPECT_THROW( (void)futureErr.get(), std::runtime_error );

    // After an exception, the pool should still accept and run new tasks
    auto future2 = pool->enqueue([] { return 123; });
    EXPECT_EQ(future2.get(), 123);  // The new task runs normally and returns 123
}

// **Pause and Resume Behavior**  
// Tests that calling pause() stops new tasks from executing until resume() is called&#8203;:contentReference[oaicite:6]{index=6}. 
// Tasks enqueueted during the paused state should not run until after resume.
TEST_F(ThreadPoolTest, PauseAndResume) {
    std::atomic<int> counter{0};

    // enqueue a couple of long-running tasks before pausing (to occupy threads)
    auto longTask = [&counter]() {
        std::this_thread::sleep_for(100ms);
        counter.fetch_add(1);
    };
    auto f1 = pool->enqueue(longTask);
    auto f2 = pool->enqueue(longTask);
    std::this_thread::sleep_for(20ms);  // give tasks a brief head start

    pool->pause();  // pause the thread pool; new tasks should not be executed now

    // enqueue tasks while the pool is paused
    auto pausedTask1 = pool->enqueue([&counter]() { counter.fetch_add(1); });
    auto pausedTask2 = pool->enqueue([&counter]() { counter.fetch_add(1); });

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
TEST_F(ThreadPoolTest, RepeatedPauseResume) {
    std::atomic<int> sum{0};
    // Perform several pause/resume cycles
    for (int cycle = 1; cycle <= 3; ++cycle) {
        pool->pause();
        // enqueue a task while paused (it will execute only after resume)
        auto fut = pool->enqueue([cycle, &sum]() { sum.fetch_add(cycle); });
        // Resume the pool to allow task execution
        pool->resume();
        // The task should be able to complete now and we get its future result
        fut.get();  // Wait for task completion
    }
    // After all cycles, the sum should equal 1+2+3 = 6 (all tasks ran exactly once)
    EXPECT_EQ(sum.load(), 6);
}

// **Shutdown Mechanics and Thread Completion**  
// Ensures that all tasks complete before the thread pool shuts down. Tests both explicit shutdown() and implicit via destructor.
TEST(ThreadPoolLifecycle, ShutdownCompletesTasks) {
    // Case 1: Implicit shutdown via destructor
    std::atomic<bool> taskRan1{false};
    {
        ThreadPool pool1(1);
        pool1.enqueue([&taskRan1]{
            std::this_thread::sleep_for(50ms);
            taskRan1.store(true);
        });
        // Going out of scope should destroy pool1; it must wait for the task to finish (taskRan1 becomes true) before thread exits.
    }
    EXPECT_TRUE(taskRan1.load()) << "Task did not complete before pool destruction";

    // Case 2: Explicit shutdown() method
    ThreadPool pool2(2);
    std::atomic<bool> taskRan2{false};
    auto fut = pool2.enqueue([&taskRan2]{
        std::this_thread::sleep_for(50ms);
        taskRan2.store(true);
    });
    pool2.shutdown();  // gracefully shut down the pool (no new tasks, wait for current tasks)
    // After shutdown returns, the enqueueted task should have completed
    EXPECT_TRUE(taskRan2.load()) << "Task did not complete before shutdown() returned";
    // Once shut down, enqueueting a new task should fail (e.g., throw an exception or return an invalid future)
    EXPECT_ANY_THROW({
        pool2.enqueue([]{ /* no-op task */ });
    });
}

// **Stress Test with High Task Volume**  
// Launches a high volume of tasks from multiple threads to stress the thread pool. 
// Verifies that all tasks are executed and none are lost or duplicated.
TEST(ThreadPoolConcurrency, HighVolumeStress) {
    const int numThreads = 4;
    const int tasksPerThread = 1000;
    const int totalTasks = numThreads * tasksPerThread;
    ThreadPool pool(8);  // use 8 worker threads for this stress test

    std::atomic<int> counter{0};
    // Launch multiple producer threads that enqueue tasks concurrently
    std::vector<std::thread> producers;
    for (int t = 0; t < numThreads; ++t) {
        producers.emplace_back([&pool, &counter, tasksPerThread]() {
            for (int i = 0; i < tasksPerThread; ++i) {
                pool.enqueue([&counter]() {
                    counter.fetch_add(1, std::memory_order_relaxed);
                });
            }
        });
    }
    // Join all producer threads (all tasks have been enqueueted)
    for (auto& pt : producers) {
        pt.join();
    }
    // Wait for all tasks to complete
    pool.waitForCompletion();
    // The counter should equal the total number of tasks enqueueted
    EXPECT_EQ(counter.load(), totalTasks);
}

// **Zero-Thread Pool Fallback Behavior**  
// If the thread pool is created with 0 threads, it should fall back to executing tasks in the calling thread (synchronously).
TEST(ThreadPoolSpecialCases, ZeroThreadsFallback) {
    ThreadPool pool(0);  // create a pool with zero worker threads
    std::atomic<bool> executed{false};
    auto fut = pool.enqueue([&executed]() { executed.store(true); return 5; });
    // Without worker threads, the task should run immediately in the enqueueter's thread.
    // The future should already be ready almost instantly.
    EXPECT_EQ(fut.wait_for(0ms), std::future_status::ready);
    EXPECT_TRUE(executed.load());
    EXPECT_EQ(fut.get(), 5);
}

// **Single-Thread FIFO Ordering**  
// Verifies that a single-threaded thread pool processes tasks in first-in-first-out order&#8203;:contentReference[oaicite:7]{index=7}.
TEST(ThreadPoolSpecialCases, SingleThreadFIFO) {
    ThreadPool singleThreadPool(1);  // only one thread
    std::vector<int> executionOrder;
    std::mutex orderMutex;

    // enqueue several tasks that record their execution sequence
    for (int i = 1; i <= 5; ++i) {
        singleThreadPool.enqueue([i, &executionOrder, &orderMutex]() {
            std::lock_guard<std::mutex> lock(orderMutex);
            executionOrder.push_back(i);
        }).get();  // use get() to wait for completion of each task (since only one thread, it runs tasks sequentially)
    }
    // The executionOrder vector should contain 1,2,3,4,5 in order of submission
    std::vector<int> expected = {1, 2, 3, 4, 5};
    EXPECT_EQ(executionOrder, expected);
}

// **Work Stealing and Nested Tasks**  
// Tests that the thread pool handles tasks which enqueue additional subtasks. 
// This scenario can cause thread-starvation deadlock if work stealing is not implemented&#8203;:contentReference[oaicite:8]{index=8}. 
// We expect the pool to execute nested tasks correctly (other threads steal the work if necessary&#8203;:contentReference[oaicite:9]{index=9}).
TEST(ThreadPoolSpecialCases, NestedTasksWorkStealing) {
    ThreadPool pool(4);
    // The outer task will enqueue two inner tasks and wait for their results
    auto outerFuture = pool.enqueue([&pool]() {
        auto inner1 = pool.enqueue([]() -> int { std::this_thread::sleep_for(10ms); return 1; });
        auto inner2 = pool.enqueue([]() -> int { std::this_thread::sleep_for(10ms); return 2; });
        // Wait for inner tasks to finish and combine results
        int result1 = inner1.get();
        int result2 = inner2.get();
        return result1 + result2;
    });
    // If work stealing is working, the inner tasks will be taken by other threads while the outer task waits.
    // This prevents deadlock and allows the outer task to complete with the correct result.
    EXPECT_EQ(outerFuture.get(), 3);
}

// **Error Handling for waitForCompletion() in Worker Thread**  
// Calling waitForCompletion (or equivalent join) from within a worker thread should be handled to avoid deadlock. 
// We expect an exception to be thrown in this scenario&#8203;:contentReference[oaicite:10]{index=10}.
TEST(ThreadPoolSpecialCases, WaitForCompletionInWorker) {
    ThreadPool pool(2);
    std::promise<bool> gotException;
    // enqueue a task that will call waitForCompletion() on the pool from within a worker
    auto fut = pool.enqueue([&pool, &gotException]() {
        try {
            pool.waitForCompletion();  // worker thread waiting for all tasks (including itself)
            gotException.set_value(false);  // if it returns (no exception), that's unexpected in a correct implementation
        } catch (...) {
            // An exception (e.g., to signal potential deadlock) is expected here
            gotException.set_value(true);
        }
    });
    // Wait for a short time to see if the promise is set (i.e., the call returned or threw)
    auto exceptionFuture = gotException.get_future();
    if (exceptionFuture.wait_for(500ms) == std::future_status::timeout) {
        FAIL() << "Deadlock: waitForCompletion called from worker thread did not return or throw in time";
    }
    EXPECT_TRUE(exceptionFuture.get()) << "waitForCompletion from inside worker did not throw an exception";
    
    // Ensure the future is resolved (get will rethrow exception if not caught inside, which in this case it was caught)
    fut.get();
}

// **Delayed Task Scheduling Timing**  
// If the thread pool supports scheduling tasks to run after a delay, verify that the task executes after the specified delay (not before). 
// We allow a tolerance on timing for thread scheduling.
TEST(ThreadPoolSpecialCases, DelayedTaskScheduling) {
    ThreadPool pool(1);
    auto startTime = std::chrono::steady_clock::now();
    // Simulate scheduling by delaying execution inside the task
    auto future = pool.enqueue([] {
        std::this_thread::sleep_for(100ms);
        return std::chrono::steady_clock::now();
    });
    // Get the actual execution time recorded by the task
    auto execTime = future.get();
    auto elapsed = execTime - startTime;
    // The task should not execute before 100ms have elapsed
    EXPECT_GE(elapsed, 100ms);
    // The task should execute not too far past the delay (allow some tolerance, e.g., < 500ms)
    EXPECT_LT(elapsed, 500ms);
}

TEST(ThreadPoolAdvanced, NestedTaskCreation) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    std::atomic<bool> allTasksCompleted{false};
    
    // A task that spawns more tasks
    auto nestedTask = [&pool, &counter]() {
        counter++;
        for (int i = 0; i < 10; i++) {
            pool.enqueue([&counter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                counter++;
            });
        }
    };
    
    // Submit several parent tasks
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 10; i++) {
        futures.push_back(pool.enqueue(nestedTask));
    }
    
    // Wait for parent tasks to complete
    for (auto& f : futures) {
        f.get();
    }
    
    // Wait for all tasks to complete
    pool.waitForCompletion();
    allTasksCompleted = true;
    
    EXPECT_TRUE(allTasksCompleted);
    EXPECT_EQ(counter.load(), 110); // 10 parent tasks + 10*10 child tasks
}

TEST(ThreadPoolAdvanced, RecursiveTaskCreation) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    
    // Define a recursive function that creates tasks up to a certain depth
    std::function<void(int)> recursiveTask = [&pool, &counter, &recursiveTask](int depth) {
        counter++;
        if (depth > 0) {
            pool.enqueue([depth, &recursiveTask]() {
                recursiveTask(depth - 1);
            });
        }
    };
    
    // Start with depth 5
    auto future = pool.enqueue([&recursiveTask]() {
        recursiveTask(5);
    });
    
    future.get();
    pool.waitForCompletion();
    
    EXPECT_EQ(counter.load(), 6); // 1 + 1 + 1 + 1 + 1 + 1 (original + 5 levels)
}

TEST(ThreadPoolPerformance, WorkStealing) {
    // Test with large number of threads to observe work stealing
    ThreadPool pool(std::thread::hardware_concurrency() * 2);
    
    std::atomic<int> completed_tasks(0);
    std::atomic<int> long_tasks_on_same_thread(0);
    const int NUM_TASKS = 1000;
    
    // Use a thread_local variable to track work distribution
    std::vector<std::future<void>> futures;
    
    for (int i = 0; i < NUM_TASKS; i++) {
        bool longTask = (i % 10 == 0); // Make every 10th task a long task
        
        futures.push_back(pool.enqueue([longTask, &completed_tasks, &long_tasks_on_same_thread]() {
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
            
            completed_tasks++;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& f : futures) {
        f.get();
    }
    
    EXPECT_EQ(completed_tasks.load(), NUM_TASKS);
    // We expect efficient work stealing to distribute long tasks
    // across different threads, so there should be few cases of
    // consecutive long tasks on the same thread
    EXPECT_LT(long_tasks_on_same_thread.load(), NUM_TASKS / 20);
}

TEST(ThreadPoolTiming, TaskTimingAccuracy) {
    ThreadPool pool(4);
    const int NUM_TIMINGS = 100;
    std::vector<double> execution_times;
    
    // Check timing of low-latency tasks
    for (int i = 0; i < NUM_TIMINGS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto future = pool.enqueue([]() {
            // Empty task to measure scheduling overhead
            return true;
        });
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
    EXPECT_LT(stdev, mean); // Standard deviation should be less than the mean
    std::cout << "Mean task scheduling time: " << mean << "ms, StdDev: " << stdev << "ms" << std::endl;
}

TEST(ThreadPoolAdvanced, DynamicWorkloadHandling) {
    ThreadPool pool(4);
    std::vector<std::future<int>> futures;
    std::atomic<int> completed{0};
    
    // Phase 1: Submit a batch of short tasks
    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed++;
            return 1;
        }));
    }
    
    // Phase 2: Submit a few long-running tasks
    for (int i = 0; i < 4; i++) {
        futures.push_back(pool.enqueue([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            completed++;
            return 10;
        }));
    }
    
    // Phase 3: Submit more short tasks that should be interleaved with long ones
    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&completed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            completed++;
            return 1;
        }));
    }
    
    int result_sum = 0;
    for (auto& fut : futures) {
        result_sum += fut.get();
    }
    
    EXPECT_EQ(completed.load(), 204);
    EXPECT_EQ(result_sum, 100 + 40 + 100);
}

TEST(ThreadPoolResilience, PauseResumeStress) {
    ThreadPool pool(4);
    std::atomic<int> processed(0);
    std::atomic<bool> keepRunning(true);
    
    // Thread that constantly toggles pause/resume
    std::thread toggle_thread([&pool, &keepRunning]() {
        int toggles = 0;
        while (keepRunning && toggles < 1000) {
            pool.pause();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            pool.resume();
            std::this_thread::sleep_for(std::chrono::microseconds(50));
            toggles++;
        }
    });
    
    // Submit a steady stream of tasks
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; i++) {
        futures.push_back(pool.enqueue([&processed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            processed++;
        }));
    }
    
    // Wait for all tasks to complete
    for (auto& fut : futures) {
        fut.get();
    }
    
    keepRunning = false;
    toggle_thread.join();
    
    EXPECT_EQ(processed.load(), 1000);
}

TEST(ThreadPoolResilience, ExceptionSafety) {
    ThreadPool pool(4);
    std::atomic<int> exceptionsHandled(0);
    std::atomic<int> normalTasksCompleted(0);
    
    // Mix of normal and exception-throwing tasks
    std::vector<std::future<void>> futures;
    
    // Submit tasks that throw exceptions
    for (int i = 0; i < 50; i++) {
        futures.push_back(pool.enqueue([i, &exceptionsHandled]() -> void {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            exceptionsHandled++;
            if (i % 2 == 0) throw std::runtime_error("Even task error");
            if (i % 3 == 0) throw std::logic_error("Divisible by 3 error");
            if (i % 5 == 0) throw 42; // Non-standard exception
        }));
    }
    
    // Submit normal tasks
    for (int i = 0; i < 100; i++) {
        futures.push_back(pool.enqueue([&normalTasksCompleted]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            normalTasksCompleted++;
        }));
    }
    
    // Verify all tasks complete without crashing the thread pool
    int exceptions = 0;
    for (auto& fut : futures) {
        try {
            fut.get();
        } catch (...) {
            exceptions++;
        }
    }
    
    // Verify thread pool still works after exceptions
    auto verification_future = pool.enqueue([]() { return 42; });
    
    EXPECT_GT(exceptions, 0);
    EXPECT_EQ(normalTasksCompleted.load(), 100);
    EXPECT_EQ(exceptionsHandled.load(), 50);
    EXPECT_EQ(verification_future.get(), 42);
}

TEST(ThreadPoolResilience, ZeroAndMaxThreads) {
    // Test with 0 threads (should immediately execute tasks in caller thread)
    {
        ThreadPool pool(0);
        bool executed = false;
        
        auto start = std::chrono::high_resolution_clock::now();
        auto fut = pool.enqueue([&executed]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            executed = true;
            return 42;
        });
        
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
        
        for (unsigned i = 0; i < std::thread::hardware_concurrency() * 10; i++) {
            futures.push_back(pool.enqueue([&counter]() {
                counter++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                return counter.load();
            }));
        }
        
        for (auto& fut : futures) {
            fut.get();
        }
        
        EXPECT_EQ(counter.load(), std::thread::hardware_concurrency() * 10);
    }
}

TEST(ThreadPoolShutdown, ShutdownUnderLoad) {
    ThreadPool pool(4);
    std::atomic<int> completed(0);
    std::vector<std::future<void>> futures;
    
    // Submit a bunch of tasks
    for (int i = 0; i < 1000; i++) {
        futures.push_back(pool.enqueue([&completed, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(i % 10 + 1));
            completed++;
        }));
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
    try {
        auto future = pool.enqueue([]() { return true; });
        FAIL() << "Pool accepted a task after shutdown";
    } catch (const std::runtime_error&) {
        // Expected
    } catch (...) {
        FAIL() << "Unexpected exception type when submitting task after shutdown";
    }
}

TEST(ThreadPoolSpecialCases, WaitForCompletionOnEmptyPool) {
    ThreadPool pool(4);
    // When the pool is idle, waitForCompletion should return quickly.
    auto start = std::chrono::steady_clock::now();
    pool.waitForCompletion();
    auto elapsed = std::chrono::steady_clock::now() - start;
    // Expect waitForCompletion to return in under 10ms on an idle pool.
    EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(), 10)
        << "waitForCompletion took too long on an idle pool";
}

TEST(ThreadPoolSpecialCases, MultipleShutdownCalls) {
    ThreadPool pool(4);
    std::atomic<int> counter{0};
    for (int i = 0; i < 10; i++) {
        pool.enqueue([&counter]{
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            counter++;
        });
    }
    // Initiate shutdown once.
    pool.shutdown();
    // A subsequent call to shutdown() should not cause any issues.
    EXPECT_NO_THROW(pool.shutdown());
    EXPECT_EQ(counter.load(), 10);
}