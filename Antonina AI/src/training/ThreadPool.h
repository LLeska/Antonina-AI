#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
public:
	ThreadPool(size_t num_threads);
	~ThreadPool();

	void enqueue(std::function<void()> task);
	void wait_all();

private:
	std::vector<std::thread> workers_;
	std::queue<std::function<void()>> tasks_;
	std::mutex queue_mutex_, done_mutex_;
	std::condition_variable cv_, done_cv_;
	std::atomic<int> active_tasks_;
	bool stop_;
};

