#pragma once

#include <streambuf>
#include <string>

class LiveStats;

class ViewerLogStreamBuf : public std::streambuf {
public:
	ViewerLogStreamBuf(LiveStats& stats);

	void flushPartial();

protected:
	int overflow(int ch) override;
	std::streamsize xsputn(const char* s, std::streamsize count) override;
	int sync() override;

private:
	void appendChar(char ch);

	LiveStats& stats_;
	std::string line_;
};

