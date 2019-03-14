#include <cstdio>
#include <string>
#include <algorithm>
#include <map>
#include <stdexcept>
#include <sstream>
#include <fstream>
using namespace std;


struct Parameter{
    
    map<string,int> readParameters(const char *filename)//filename表示配置文件的名字
    {
        map<string,int>ParametersMap;
        string line;
        string paramName;
        int paramValue = 0;
        string paramValuestr;
        ifstream fin(filename);//设置文件流
    
        if(!fin.good()){//如果文件流读取失败则返回异常
            string msg("Parameters file not found!");
            msg.append("filename");
            throw runtime_error(msg);
        }
    
    while(fin.good()){
        getline(fin,line);//从文件流中提取一行赋值给line
        stringstream buffer;
        buffer << line;
        buffer >> paramName;
        //printf("paramName=%s\n",paramName.c_str());
        if(paramName.compare("dimension:")==0){
            buffer >> paramValue;
            ParametersMap["dimension"]=paramValue;
        }
        else if(paramName.compare("M:")==0){
            buffer >> paramValue;
            ParametersMap["M"]=paramValue;
        }
        else if(paramName.compare("efCon:")==0){
            buffer >> paramValue;
            ParametersMap["efCon"]=paramValue;
        }
        else if(paramName.compare("efSearch:")==0){
            buffer >> paramValue;
            ParametersMap["efSearch"]=paramValue;
        }
        else if(paramName.compare("verbose:")==0){
            buffer >> paramValue;
            ParametersMap["verbose"]=paramValue;
        }
        else if(paramName.compare("metric_type:")==0){
            buffer >> paramValue;
            ParametersMap["metric_type"]=paramValue;
        }
        else if(paramName.compare("topk:")==0){
            buffer >> paramValue;
            ParametersMap["topk"]=paramValue;
        }
        else if(paramName.compare("num_threads:")==0){
            buffer >> paramValue;
            ParametersMap["num_threads"]=paramValue;
        }
        else if(paramName.compare("QuantizerType:")==0){
            buffer >> paramValue;
            ParametersMap["QuantizerType"]=paramValue;
            printf("QuantizerType=%d\n",paramValue);
        }
        else
        {
            throw runtime_error(string("unknown Parameter:").append(paramName));
        }
    }
    fin.close();
    return ParametersMap;

    }
};

