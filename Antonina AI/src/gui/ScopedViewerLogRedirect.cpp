#include "ScopedViewerLogRedirect.h"

ScopedViewerLogRedirect::ScopedViewerLogRedirect(std::ostream& stream, LiveStats& stats) : stream_(stream), buffer_(stats), old_(stream.rdbuf(&buffer_)) {
}

ScopedViewerLogRedirect::~ScopedViewerLogRedirect() {
    buffer_.flushPartial();
    stream_.rdbuf(old_);
}
