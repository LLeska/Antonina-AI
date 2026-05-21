#pragma once

struct TrainingSettings;

const char *settingsFileName();
bool loadSettingsJson(TrainingSettings &settings);
bool saveSettingsJson(const TrainingSettings &settings);
