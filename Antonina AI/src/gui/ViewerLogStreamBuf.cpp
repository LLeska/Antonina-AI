#include "ViewerLogStreamBuf.h"
#include "LiveStats.h"

ViewerLogStreamBuf::ViewerLogStreamBuf(LiveStats& stats) : stats_(stats) {}

void ViewerLogStreamBuf::flushPartial() {
    if (!line_.empty()) {
        stats_.appendLog(line_);
        line_.clear();
    }
}

int ViewerLogStreamBuf::overflow(int ch) {
    if (ch == traits_type::eof())
        return traits_type::not_eof(ch);
    appendChar((char)ch);
    return ch;
}

std::streamsize ViewerLogStreamBuf::xsputn(const char* s,
    std::streamsize count) {
    for (std::streamsize i = 0; i < count; ++i)
        appendChar(s[i]);
    return count;
}

int ViewerLogStreamBuf::sync() { return 0; }

void ViewerLogStreamBuf::appendChar(char ch) {
    if (ch == '\n' || ch == '\r') {
        flushPartial();
        return;
    }
    if (line_.size() < 1200)
        line_.push_back(ch);
}

