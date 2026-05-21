#include "ThreadPool.h"


ThreadPool::ThreadPool(size_t num_threads) : stop_(false), active_tasks_(0) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty())
                        return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
                {
                    std::unique_lock<std::mutex> lock(done_mutex_);
                    --active_tasks_;
                }
                done_cv_.notify_one();
            }
            });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& worker : workers_)
        worker.join();
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(done_mutex_);
        ++active_tasks_;
    }
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        tasks_.push(std::move(task));
    }
    cv_.notify_one();
}

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [this] { return active_tasks_ == 0; });
}
