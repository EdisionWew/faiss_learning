#include <cstdio>
#include <cstdlib>
#include <vector>
#include <faiss/IndexHNSW.h>
#include "HNSW_FLATSQ.h"
#include <faiss/IndexFlat.h>
#include <faiss/Index.h>
using namespace std;
int main()
{
    
    /***********************************数据载入***************************************/
    int d = 64;                            // dimension
    int nb = 1000;                       // database size
    int nq = 10;                        // nb of queries
    
    

    float *xb = new float[d * nb];//申请100000x64的数据空间
    float *xq = new float[d * nq];//申请10000x64的查询数据空间

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();//使用随机数填充生成的数据空间
        xb[d * i] += i / 10.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();//使用随机数填充生成的数据空间
        xq[d * i] += i / 10.;
    }
    

    /***********************************准备参数**************************************/
    
    int indexNum = 10;//批量生成hnsw_Flat_Index的个数
    
    string filename = "./hnsw_Flat.conf";//配置文件的绝对路径
    
    struct HNSW_FLAT hnsw_flat(filename.c_str());
    
    int topk=hnsw_flat.topk;//检索最相似向量的个数
    
    /************************************批量生成Index*************************************/
    
    vector<faiss::IndexHNSWSQ*> Indexs;
    hnsw_flat.getIndex(indexNum,&Indexs);//批量生成Index接口
    
    /************************************测试生成Index*************************************/
    
    Indexs[0]->train(nb,xb);
    Indexs[0]->add(nb,xb);
    long *I = new long[topk * nq];//检索出相似向量的id
    float *D = new float[topk * nq];
    
    Indexs[0]->search(nq,xq,topk,D,I);
    
        
    for(int i=0;i<nq;i++)
    {
        for(int j=0;j<topk;j++)
        {
            printf("%5ld ",I[i*topk+j]);
        }
        printf("\n");
    }
    
    /************************************释放内存*************************************/
   
    /*
    Grab *grab=new Grab(d,xb,xq,nb,nq);
    grab->Index_Flat(4);
    vector<Result_Grab>::iterator it=grab->RESULT.begin();
    for(int i = 0; i<nq;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ",(it+i*4+j)->id);
        }
        printf("\n");
    }
    */
    delete [] xb;
    delete [] xq;
    return 0;
}
