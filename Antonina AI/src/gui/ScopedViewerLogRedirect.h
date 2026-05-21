#pragma once

#include "ViewerLogStreamBuf.h"

#include <ostream>


class ScopedViewerLogRedirect {
public:
	ScopedViewerLogRedirect(std::ostream& stream, LiveStats& stats);
	~ScopedViewerLogRedirect();

private:
	std::ostream& stream_;
	ViewerLogStreamBuf buffer_;
	std::streambuf* old_;
};
