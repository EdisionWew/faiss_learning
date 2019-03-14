# faiss-这是基于faiss库的两个类编写的接口
HNSW_FLATSQ.h  是生成IndexHNSWFlat 和 IndexHNSWSQ 两种index的接口文件


类HNSW_FLAT

	HNSW_FLAT(const char* configfilename)
	//类HNSW_FLAT的构造函数 configfilename是包含参数的配置文件绝对路径

	void getIndex(int indexNum,vector<IndexHNSWFlat*> *Indexs)
	//该函数是批量生成IndexHNSWFlat的接口函数，indexNum表示要生成的index的个数 indexs是生成的index的集合
	
	void getIndex(int indexNum,vector<IndexHNSWSQ*> *Indexs)
	//该函数是批量生成IndexHNSWSQ的接口函数，indexNum表示要生成的index的个数 indexs是生成的index的集合


注意：每个index生成后
	  添加数据使用add(long int n, const float *x)
	  训练数据使用train(long int n, const float *x)
	  检索数据使用search(long int n, const float *x, const float *distance, long int id)


hnsw_Flat.conf是包含各个参数的配置文件


demo_main.cpp是index检索接口实例化的demo

	
