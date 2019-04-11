/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <iostream>
#include <queue>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <faiss/index_io.h>
#include <sys/time.h>
#include <vector>
#include "../AutoTune.h"
using namespace std;

//static omp_lock_t lock;

typedef faiss::Index::idx_t idx_t;
typedef struct Node 
{
      float value;
      faiss::Index::idx_t idx;
      //Node (float v, faiss::Index::idx_t i): value(v), idx(i) {}
      //friend bool operator < (const Node &n1, const Node &n2) ;
}Node;//构造结果距离、id节点

int cmp(const Node a,const Node b)
{
    if(a.value<b.value)
        return 1;
    else
        return 0;
}

//inline bool operator < (const Node &n1, const Node &n2) 
 //{
//      return -n1.value < -n2.value;
// }

/*****************************************************
 * I/O functions for fvecs and ivecs
 *****************************************************/


float * fvecs_read (const char *fname,
                    size_t *d_out, size_t *n_out)
{
    FILE *f = fopen(fname, "r");
    if(!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) || !"unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = st.st_size;
    assert(sz % ((d + 1) * 4) == 0 || !"weird file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = d; *n_out = n;
    float *x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) || !"could not read whole file");

    // shift array to remove row headers
    for(size_t i = 0; i < n; i++)
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));

    fclose(f);
    return x;
}

// not very clean, but works as long as sizeof(int) == sizeof(float)
int *ivecs_read(const char *fname, size_t *d_out, size_t *n_out)
{
    return (int*)fvecs_read(fname, d_out, n_out);
}

double elapsed ()
{
    struct timeval tv;
    gettimeofday (&tv, nullptr);
    return  tv.tv_sec + tv.tv_usec * 1e-6;
}

int main()
{
    double t0 = elapsed();
    size_t nq;
    float *xq;

    {
        printf ("[%.3f s] Loading queries\n", elapsed() - t0);

        size_t d2;
        xq = fvecs_read("sift1M/sift_query.fvecs", &d2, &nq);
//         assert(d == d2 || !"query does not have same dimension as train set");

    }//读取查询数据

    size_t k; // nb of results per query in the GT
    faiss::Index::idx_t *gt;  // nq * k matrix of ground-truth nearest-neighbors

    {
        printf ("[%.3f s] Loading ground truth for %ld queries\n",
                elapsed() - t0, nq);

        // load ground-truth and convert int to long
        size_t nq2;
        int *gt_int = ivecs_read("sift1M/sift_groundtruth.ivecs", &k, &nq2);
        assert(nq2 == nq || !"incorrect nb of ground truth entries");

        gt = new faiss::Index::idx_t[k * nq];
        for(int i = 0; i < k * nq; i++) {
            gt[i] = gt_int[i];
        }
        delete [] gt_int;
    }//读取ground truth	文件
    
    vector<faiss::Index *> indexes;
    int index_total_num = 10;
    {
                
        string filename="HNSW32_SQ8.index";
        for(int i = 0;i < index_total_num;i ++)
        {
            cout<<i<<endl;
            faiss::Index * index = faiss::read_index(filename.c_str());
            indexes.push_back(index);
        }//读取index文件，批量生成index（注：这里每个测试index里的数据是一样的，使用时可以不一样）
        
        // output buffers
        faiss::Index::idx_t *I = new  faiss::Index::idx_t[nq * k*index_total_num];
        float *D = new float[nq * k*index_total_num];
        
        for(int i = 0;i < index_total_num;i ++)
        {
            indexes[i]->search(nq, xq, k, D+i*k*nq, I+i*k*nq); 
            
        }//检索数据并获取结果
        
        
        float * newValue=new float[nq*k];
        idx_t * newID=new idx_t[nq*k];
       
        printf ("[%.3f s] merge result----->Start\n", elapsed() - t0);
        // merge result
        
        
        
        int nb_thread=10;//设置线程数
        omp_set_num_threads(nb_thread);
		vector<vector<Node> > pqs;
        for(int i=0;i<nq;i++)
        {
            vector<Node> sd(index_total_num*k);
            pqs.push_back(sd);
        }
        int per_index=index_total_num/nb_thread;//分配每个线程的处理index数目
        #pragma omp parallel//设立并行区域------收集阶段
            {
                
                int id_thread=omp_get_thread_num();
                if(id_thread==nb_thread-1)
                {
                    for(int i=id_thread*per_index;i< index_total_num;i++)
                    {
                        for(int j = 0;j < nq;j ++)
                        {
                            for(int kk = 0,start=id_thread*k;kk < k;kk ++,start++)
                            { 
                                Node tou;
                                tou.idx=I[i*nq*k+j*k+kk]+i*1000000;
                                tou.value=D[i*nq*k+j*k+kk];
                                pqs[j][start]=tou;
                        
                            }
                        }
                    }
                }
                else
                {
                    for(int i=id_thread*per_index;i< (id_thread+1)*per_index;i++)
                    {
                        for(int j = 0;j < nq;j ++)
                        {
                            for(int kk = 0,start=id_thread*k;kk < k;kk ++,start++)
                            { 
                                Node tou;
                                tou.idx=I[i*nq*k+j*k+kk]+i*1000000;
                                tou.value=D[i*nq*k+j*k+kk];
                                pqs[j][start]=tou;
                        
                            }
                        }
                    }
                }
                
                
                #pragma omp barrier  // 排序阶段
                int nb_nq_per_threads=nq/nb_thread;
                if(id_thread==nb_thread-1)
                {
                    for(int i=id_thread*nb_nq_per_threads;i< nq;i++)
                    {
                        sort(pqs[i].begin(),pqs[i].end(),cmp);
                    }
                }
                else
                {
                    for(int i=id_thread*nb_nq_per_threads;i< (id_thread+1)*nb_nq_per_threads;i++)
                    {
                        sort(pqs[i].begin(),pqs[i].end(),cmp);
                    }
                }
                
                
            }
            
		for(int j = 0;j < nq;j ++)
		{
			for(int kk =0;kk < k;kk ++)
			{
				newValue[j*k+kk] = pqs[j][kk].value;
				newID[j*k+kk] = pqs[j][kk].idx;
			}
		}
        printf ("[%.3f s] merge result----->end\n", elapsed() - t0);
        //重排结果并返回最终topk的结果
        
        
        printf ("[%.3f s] Compute recalls\n", elapsed() - t0);

        // evaluate result by hand.
        int n_1 = 0, n_10 = 0, n_100 = 0;
        for(int i = 0; i < nq; i++) {
            int gt_nn = gt[i * k];
            for(int j = 0; j < k; j++) {
                if (newID[i * k + j] == gt_nn) {
                    if(j < 1) n_1++;
                    if(j < 10) n_10++;
                    if(j < 100) n_100++;
                }
            }
        }//计算检索结果召回率

        printf("R@1 = %.4f\n", n_1 / float(nq));
        printf("R@10 = %.4f\n", n_10 / float(nq));
        printf("R@100 = %.4f\n", n_100 / float(nq));
        
        delete [] I;
        delete [] D;
        delete [] newID;
        delete [] newValue;//释放对象空间

    }
    
    //
    

    delete [] xq;
    delete [] gt;
    for(int i = 0;i < index_total_num;i ++)
    {
        delete indexes[i];        
    }
    //omp_destroy_lock(&lock);
    return 0;
}
