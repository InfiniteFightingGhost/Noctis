#ifndef API_CLIENT_H
#define API_CLIENT_H

#include <Arduino.h>
#include "types.h"

void setupAPIClient();
void setRecordingID(String id);
void setAPIEndpoint(String endpoint);
void setAPIKey(String key);
String getAPIEndpoint();
String getAPIKey();
void pingAPI();

String startNewRecording();
bool uploadChunkAsEpochBatch(DataChunk* chunk, uint32_t chunkIndex);
void uploadBacklog();

// For serial commands
void printAPIStatus();
uint32_t getBacklogCount();

#endif // API_CLIENT_H
