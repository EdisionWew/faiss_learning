#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>
#include <map>
#include <faiss/IndexHNSW.h>
#include "Parameters.h"
#include <faiss/Index.h>
#include <faiss/IndexScalarQuantizer.h>
using namespace std;
struct HNSW_FLAT
{
    typedef faiss::Index::idx_t idx_t;
    typedef faiss::IndexHNSWFlat IndexHNSWFlat;
    typedef faiss::IndexHNSWSQ IndexHNSWSQ;
    typedef faiss::ScalarQuantizer::QuantizerType QuantizerType;
    map<string,int> parameters;
    idx_t topk;
    idx_t num_threads;
    explicit HNSW_FLAT(const char* configfilename)
    {
        struct Parameter pa;
        parameters=pa.readParameters(configfilename);//获取参数
        topk=parameters["topk"];
        num_threads=parameters["num_threads"];
    }
    
     void getIndex(int indexNum,vector<IndexHNSWFlat*> *Indexs)
     {
        for(int i=0;i<indexNum;i++)
        {
            IndexHNSWFlat *index=new IndexHNSWFlat((idx_t)parameters["dimension"],(idx_t)parameters["M"]);
            index-> hnsw.efConstruction=parameters["efCon"];
            index-> hnsw.efSearch=parameters["efSearch"];
            if(parameters["metric_type"]==0)
                index->metric_type=faiss::METRIC_INNER_PRODUCT;
            else
                index->metric_type=faiss::METRIC_L2; 
            index->verbose=parameters["verbose"];
            Indexs->push_back(index);//把申请的index压入Indexs
        }
    }
    
    void getIndex(int indexNum,vector<IndexHNSWSQ*> *Indexs)
    {
        for(int i=0;i<indexNum;i++)
        {
            IndexHNSWSQ *index;
            if(parameters["QuantizerType"]==0)
                index=new IndexHNSWSQ((idx_t)parameters["dimension"],QuantizerType::QT_8bit,(idx_t)parameters["M"]);
            else if(parameters["QuantizerType"]==1)
                index=new IndexHNSWSQ((idx_t)parameters["dimension"],QuantizerType::QT_4bit,(idx_t)parameters["M"]);
            else if(parameters["QuantizerType"]==2)
                index=new IndexHNSWSQ((idx_t)parameters["dimension"],QuantizerType::QT_8bit_uniform,(idx_t)parameters["M"]);
            else if(parameters["QuantizerType"]==3)
                index=new IndexHNSWSQ((idx_t)parameters["dimension"],QuantizerType::QT_4bit_uniform,(idx_t)parameters["M"]);
            else
                index=new IndexHNSWSQ((idx_t)parameters["dimension"],QuantizerType::QT_fp16,(idx_t)parameters["M"]);
            index-> hnsw.efConstruction=parameters["efCon"];
            index-> hnsw.efSearch=parameters["efSearch"];
            if(parameters["metric_type"]==0)
                index->metric_type=faiss::METRIC_INNER_PRODUCT;
            else
                index->metric_type=faiss::METRIC_L2; 
            index->verbose=parameters["verbose"];
            Indexs->push_back(index);//把申请的index压入Indexs
        }
    }
    
};



