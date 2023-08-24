#include <map>
#include <iostream>
#include <set>
#include <atomic>
#include <algorithm>
#include "MapReduceFramework.h"
#include "Barrier.h"


#define  START 0
#define PERCENTAGE 100
#define MUTEX_LOCK_ERROR "system error: couldnt lock mutex"
#define MUTEX_UNLOCK_ERROR "system error: couldnt unlock mutex"
#define CREATE_ERROR "system error: couldnt create pthreads"
#define DESTROY_ERROR "system error: couldnt destroy mutex"



struct ThreadContext;

struct JobContext {
    JobState job_state;
    const MapReduceClient *client;
    int num_of_int_vecs;
    Barrier *barrier;
    pthread_t *threads;
    ThreadContext *threadContexts;
    bool wait_flag;
    std::atomic<unsigned long> map_index;
    std::atomic<unsigned long> map_percentage;
    std::atomic<unsigned long> shuffleVecs;
    std::atomic<unsigned long> shuffle_percentage;
    std::atomic<unsigned long> emit2Counter;
    std::atomic<unsigned long> reduceVecs;
    std::atomic<unsigned long> reduce_percentage;
    pthread_mutex_t emit2Mutex;
    pthread_mutex_t emit3Mutex;
    pthread_mutex_t stateMutex;
    pthread_mutex_t waitMutex;
    const InputVec *inputVec;
    OutputVec *outputVec;
    IntermediateVec **allThreads_intVectors;
    std::vector<IntermediateVec *> *trade_center;

};

struct ThreadContext {
    int threadID;
    JobContext *job_context;
};

void Map(ThreadContext *thread) {
  JobContext *jobContext = thread->job_context;
  size_t elem_num = jobContext->inputVec->size();
  unsigned long prev_ind;
  while ((prev_ind = jobContext->map_index.fetch_add(1)) < elem_num) {
    InputPair k1_v1 = (*jobContext->inputVec)[prev_ind];
   jobContext->client->map(k1_v1.first, k1_v1.second, thread);
    jobContext->map_percentage.fetch_add(1);
  }
}


bool compare_func(const IntermediatePair& pair1, const IntermediatePair&
pair2) {
  return *(pair1.first) < *(pair2.first);
}


void Sort(ThreadContext *thread)
{
  auto thread_ind = thread->job_context->allThreads_intVectors[thread->threadID];
  std::sort(thread_ind->begin(),thread_ind->end (),compare_func);
}

K2* find_max(JobContext* jobContext){
    K2* max = nullptr;
    for (int ind = 0; ind < jobContext->num_of_int_vecs; ind++) {
        if (!jobContext->allThreads_intVectors[ind]->empty()) {
            K2 *currentKey = jobContext->allThreads_intVectors[ind]->back().first;
            if (max == nullptr || (!(*currentKey < *max))) {
                max = currentKey;
            }
        }
    }
    return max;
}

void Shuffle(JobContext* jobContext) {
    K2 *max;
    while ((max = find_max(jobContext)) != nullptr) {
        auto *new_vec = new IntermediateVec();
        jobContext->shuffleVecs++;
        jobContext->trade_center->push_back(new_vec);
        for (int ind = 0; ind < jobContext->num_of_int_vecs; ind++) {
            IntermediateVec *threadVec = jobContext->allThreads_intVectors[ind];
            while (!threadVec->empty() && !(*threadVec->back().first < *max)) {
                new_vec->push_back(threadVec->back());
                threadVec->pop_back();
                jobContext->shuffle_percentage++;
            }
        }
    }
}


void Reduce(JobContext *jobContext) {
    size_t elem_num = jobContext->shuffleVecs;
    unsigned long prev_ind;
    while ((prev_ind = jobContext->reduceVecs.fetch_add(1)) < elem_num) {
        IntermediateVec *queue_vec = jobContext->trade_center->at(prev_ind);
        jobContext->client->reduce(queue_vec, jobContext);
        jobContext->reduce_percentage += queue_vec->size();
    }
}

void thread_zero(JobContext *job){
  if (pthread_mutex_lock(&job->stateMutex) != 0){
    std::cout << MUTEX_LOCK_ERROR << std::endl;
    exit(1);
  }
  job->job_state.stage = SHUFFLE_STAGE;
  job->job_state.percentage = START;
  if (pthread_mutex_unlock(&job->stateMutex) != 0){
    std::cout << MUTEX_UNLOCK_ERROR << std::endl;
    exit(1);
  }
  Shuffle(job);
  if (pthread_mutex_lock(&job->stateMutex) != 0){
    std::cout << MUTEX_LOCK_ERROR << std::endl;
    exit(1);
  }
  job->job_state.stage = REDUCE_STAGE;
  job->job_state.percentage = START;
  if (pthread_mutex_unlock(&job->stateMutex) != 0){
    std::cout << MUTEX_UNLOCK_ERROR << std::endl;
    exit(1);
  }
}

void *threadCycle(void *tc) {
    auto *thread = (ThreadContext*) tc;
    JobContext *job = thread->job_context;
    if (pthread_mutex_lock(&job->stateMutex) != 0){
        std::cout << MUTEX_LOCK_ERROR << std::endl;
        exit(1);
   }
    job->job_state.stage = MAP_STAGE;
    job->job_state.percentage = START;
    if (pthread_mutex_unlock(&job->stateMutex) != 0){
        std::cout << MUTEX_UNLOCK_ERROR<< std::endl;
        exit(1);
    }
    Map(thread);
    Sort(thread);
    job->barrier->barrier();
    if (thread->threadID == 0) {
      thread_zero(job);
    }
    job->barrier->barrier();
    Reduce(job);
    return nullptr;
}

void emit2(K2 *key, V2 *value, void *context){ //todo: should i lock it?
  auto *thread = (ThreadContext*) context;
  if (pthread_mutex_lock(&thread->job_context->emit2Mutex) != 0){
        std::cout << MUTEX_LOCK_ERROR << std::endl;
        exit(1);
  }
  IntermediatePair cur_pair;
  cur_pair.first = key;
  cur_pair.second = value;
  IntermediateVec *interVec = thread->job_context->allThreads_intVectors[thread->threadID];
  thread->job_context->emit2Counter++;
  interVec->push_back(cur_pair);
  if (pthread_mutex_unlock(&thread->job_context->emit2Mutex) != 0){
        std::cout << MUTEX_UNLOCK_ERROR << std::endl;
        exit(1);
  }
}


void emit3(K3 *key, V3 *value, void *context) {
  auto *job_context = (JobContext *) context;
  if (pthread_mutex_lock(&job_context->emit3Mutex) != 0){
    std::cout << MUTEX_LOCK_ERROR << std::endl;
    exit(1);
  }
  OutputPair cur_pair;
  cur_pair.first = key;
  cur_pair.second = value;
  job_context->outputVec->push_back(cur_pair);
  if (pthread_mutex_unlock(&job_context->emit3Mutex) != 0){
    std::cout << MUTEX_UNLOCK_ERROR << std::endl;
    exit(1);
  }
}


JobHandle startMapReduceJob(const MapReduceClient &client,
                            const InputVec &inputVec, OutputVec &outputVec,
                            int multiThreadLevel) {
  auto *job_context = new JobContext();
  pthread_mutex_lock(&job_context->stateMutex);
  job_context->job_state = {UNDEFINED_STAGE, 0};
  pthread_mutex_unlock(&job_context->stateMutex);
  job_context->client = &client;
  job_context->num_of_int_vecs = multiThreadLevel;
  job_context->barrier = new Barrier(multiThreadLevel);
  job_context->threads = new pthread_t[multiThreadLevel];
  job_context->threadContexts = new ThreadContext[multiThreadLevel];
  job_context->wait_flag = false;
  job_context->map_index = START;
  job_context->map_percentage = START;
  job_context->shuffleVecs = START;
  job_context->shuffle_percentage = START;
  job_context->emit2Counter = START;
  job_context->reduceVecs = START;
  job_context->reduce_percentage = START;
  pthread_mutex_t emit2Mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t emit3Mutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t stateMutex = PTHREAD_MUTEX_INITIALIZER;
  pthread_mutex_t waitMutex = PTHREAD_MUTEX_INITIALIZER;
  job_context->inputVec = &inputVec;
  job_context->allThreads_intVectors = new IntermediateVec *[multiThreadLevel];
  for (int ind = 0; ind < multiThreadLevel; ind++) {
    job_context->allThreads_intVectors[ind] = new IntermediateVec();

  }
  job_context->trade_center = new std::vector<IntermediateVec *>();
  job_context->outputVec = &outputVec;
  for (int ind = 0; ind < multiThreadLevel; ++ind) {
    job_context->threadContexts[ind] = {ind, job_context};
    if (pthread_create(job_context->threads + ind, NULL, threadCycle,
                       job_context->threadContexts + ind) != 0){
      std::cout << CREATE_ERROR << std::endl;
      exit(1);
    }
  }
  auto *job_handle= (JobHandle *) job_context;
  return job_handle;
}



void getJobState(JobHandle job, JobState *state) {
  auto *jobContext = (JobContext*)(job);
  if (pthread_mutex_lock(&jobContext->stateMutex) != 0){
    std::cout << MUTEX_LOCK_ERROR << std::endl;
    exit(1);
  }
  if(jobContext->job_state.stage == MAP_STAGE){
    float percentage = ((float) jobContext->map_percentage/ (float)
        jobContext->inputVec->size()) * PERCENTAGE;
    *state = {MAP_STAGE, percentage};
  }
  if(jobContext->job_state.stage == SHUFFLE_STAGE)
  {
    float percentage = ((float) jobContext->shuffle_percentage / (float)
        jobContext->emit2Counter) * PERCENTAGE;
    *state = {SHUFFLE_STAGE, percentage};

  }
  if(jobContext->job_state.stage == REDUCE_STAGE)
  {
    float percentage = ((float) jobContext->reduce_percentage
                        / (float) jobContext->shuffle_percentage) * PERCENTAGE;
    *state = {REDUCE_STAGE, percentage};
  }
  if(jobContext->job_state.stage == UNDEFINED_STAGE){
    *state = {UNDEFINED_STAGE, START};
  }
  if (pthread_mutex_unlock(&jobContext->stateMutex) != 0){
    std::cout << MUTEX_UNLOCK_ERROR << std::endl;
    exit(1);
  }
}


void waitForJob(JobHandle job) {
  JobContext* jobContext = ((JobContext*) job);
  if (pthread_mutex_lock(&jobContext->waitMutex) != 0){
    std::cout << MUTEX_LOCK_ERROR << std::endl;
    exit(1);
  }
  if(!jobContext->wait_flag){
    for (int ind = 0; ind < jobContext->num_of_int_vecs; ind++) {
      pthread_join(jobContext->threads[ind], NULL); //todo: join worked?
    }
    jobContext->wait_flag = true;
  }
  if (pthread_mutex_unlock(&jobContext->waitMutex) != 0){
    std::cout << MUTEX_UNLOCK_ERROR << std::endl;
    exit(1);
  }
}


void valid_destroy(int check_input){
    if (check_input != 0) {
        std::cout << DESTROY_ERROR << std::endl;
        exit(1);
    }
}


void closeJobHandle(JobHandle job) {
    auto *jobContext = (JobContext*)(job);
    waitForJob(jobContext);
    valid_destroy(pthread_mutex_destroy(&jobContext->stateMutex));
    valid_destroy(pthread_mutex_destroy(&jobContext->emit2Mutex));
    valid_destroy(pthread_mutex_destroy(&jobContext->emit3Mutex));
    valid_destroy(pthread_mutex_destroy(&jobContext->waitMutex));
    for (IntermediateVec *vec : *(jobContext->trade_center)) {
        delete vec;
    }
    for (int ind = 0; ind < jobContext->num_of_int_vecs; ind++) {
        delete jobContext->allThreads_intVectors[ind];
    }
    delete[] jobContext->threads;
    delete[] jobContext->threadContexts;
    delete[] jobContext->allThreads_intVectors;
    delete jobContext->trade_center;
    delete jobContext->barrier;
    delete jobContext;
}

