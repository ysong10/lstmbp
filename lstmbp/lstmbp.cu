/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//		Author:    Yang Song
//		File:      CUDA implementation of LSTM model including both feed-forward and back-propagation 
///////////////////////////////////////////////////////////////////////////////////////// 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <assert.h>
using namespace std;

#define input_0 0.98
#define input_1 0.88
#define alpha 0.1

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned g_verbose;
unsigned NUM;

/////////////////////////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
		for (i=2; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;

	if (commandline_error || !NUM) {
		printf("Usage: ./LSTM <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}


	NeuralNetwork();
	printf("success\n");
       
    //CUT_EXIT(argc, argv);
}

void InitHostMem(double *w_i, double *u_i, double *b_i, double *w_f, double *u_f, double *b_f, double *w_c, double *u_c, double *b_c, double *w_o, double *u_o, double *b_o, double *weight, double *bias)
{
	// Input Gate Weights and Bias
	FILE * pFile1 = fopen ("data/w_i.txt","rb");
	if (pFile1 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile1);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_i[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile1);
	}	
	
	if (!pFile1)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile2 = fopen ("data/b_i.txt","rb");
	if (pFile2 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile2);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_i[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile2);
	}	
	//cout<<"Input gate reading completed, "<<b_i[99]<<endl;
	if (!pFile2)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Forget gate Weights and Bias
	FILE * pFile3 = fopen ("data/w_f.txt","rb");
	if (pFile3 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile3);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_f[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile3);
	}	
	
	if (!pFile3)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile4 = fopen ("data/b_f.txt","rb");
	if (pFile4 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile4);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_f[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile4);
	}	
	//cout<<"Forget gate reading completed, "<<b_f[99]<<endl;
	if (!pFile4)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Cell gate Weights and Bias
	FILE * pFile5 = fopen ("data/w_c.txt","rb");
	if (pFile5 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile5);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_c[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile5);
	}	
	
	if (!pFile5)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile6 = fopen ("data/b_c.txt","rb");
	if (pFile6 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile6);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_c[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile6);
	}	
	//cout<<"Forget gate reading completed, "<<b_c[99]<<endl;
	if (!pFile6)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Output gate Weights and Bias
	FILE * pFile7 = fopen ("data/w_o.txt","rb");
	if (pFile7 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile7);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_o[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile7);
	}	
	
	if (!pFile7)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile8 = fopen ("data/b_o.txt","rb");
	if (pFile8 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile8);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_o[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile8);
	}	
	//cout<<"Output gate reading completed, "<<b_o[99]<<endl;
	if (!pFile8)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile9 = fopen ("data/W.txt","rb");
	if (pFile9 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile9);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			weight[i] = temp_num;
			i++;
			index++;
			if(i==101)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile9);
	}
	*bias = weight[100];	
	//cout<<"Fully Connected layer reading completed, "<<*bias<<endl;
	if (!pFile9)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of output gate
	FILE * pFile10 = fopen ("data/u_o.txt","rb");
	if (pFile10 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile10);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_o[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile10);
	}	
	
	if (!pFile10)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of input gate
	FILE * pFile11 = fopen ("data/u_i.txt","rb");
	if (pFile11 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile11);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_i[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile11);
	}	
	
	if (!pFile11)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of cell gate
	FILE * pFile12 = fopen ("data/u_c.txt","rb");
	if (pFile12 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile12);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_c[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile12);
	}	
	
	if (!pFile12)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of forget gate
	FILE * pFile13 = fopen ("data/u_f.txt","rb");
	if (pFile13 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile13);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_f[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile13);
	}	
	
	if (!pFile13)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
}

__device__ double hard_sigmoid(double x)
{
	if(x<-2.5)
		return 0;
	else
	{
		if(x>2.5)
			return 1;
		else
			return (0.2*x + 0.5);
	}
}

__device__ double dsigmoid(double y)
{
	return  y * (1.0 - y);
}

__device__ double dtanh( double y)
{
	y = tanh(y);
	return 1.0 - y * y;
	
}
 
__global__ void ExecuteLSTM( double result_0, double result_1, double *w_i, double *u_i, double *b_i, double *w_f, double *u_f, double *b_f, double *w_c, double *u_c, double *b_c, double *w_o, double *u_o, double *b_o, double *weight, double *bias, double *LSTM_results, double *at_0, double *at_1, double *it_0, double *it_1, double *ft_0, double *ft_1, double *ot_0, double *ot_1, double *statet_0, double *statet_1, double *u_0, double* u_1)
{
	int x = threadIdx.x;
	double c = 0;
	double i[2];
	double temp_ua = 0, temp_ui = 0, temp_uf = 0, temp_uo = 0;
	//int index = threadIdx.x*2;
	i[0] = input_0;
	i[1] = input_1;


	at_0[x] = tanh(i[0]*w_c[x] + b_c[x]);
	it_0[x] = hard_sigmoid(i[0]*w_i[x] + b_i[x]);
        ft_0[x] = hard_sigmoid(i[0]*w_f[x] + b_f[x]);
	ot_0[x] = hard_sigmoid(i[0]*w_o[x] + b_o[x]);
	statet_0[x] = at_0[x]*it_0[x] + ft_0[x]*c;
	c = statet_0[x];
	u_0[x] = ot_0[x]*tanh(statet_0[x]);
	
	__syncthreads();
	if(x==0)
	{
		double result_0 = *bias;
		for(int i=0; i<100; i++)
		{
			result_0 += weight[i]*u_0[i];
		}
		//printf("The result for i=0 is %f\n",result);
	}
	__syncthreads();
	for(int i=0; i<100; i++)
	{
		temp_ua += u_0[i] * u_c[i*100 + x];
		temp_ui += u_0[i] * u_i[i*100 + x];
		temp_uf += u_0[i] * u_f[i*100 + x];
		temp_uo += u_0[i] * u_o[i*100 + x]; 	
	}
	
	at_1[x] = tanh(i[1]*w_c[x] + temp_ua + b_c[x]);//新记忆
	it_1[x] = hard_sigmoid(i[1]*w_i[x] + temp_ui + b_i[x]);
	ft_1[x] = hard_sigmoid(i[1]*w_f[x] + temp_uf + b_f[x]);
	ot_1[x] = hard_sigmoid(i[1]*w_o[x] + temp_uo + b_o[x]);
	
	statet_1[x]= at_1[x]*it_1[x] + ft_1[x]*c; //最终记忆
	u_1[x] = ot_1[x]*tanh(statet_1[x]);
	//隐藏层输出
	__syncthreads();
	if(x==0)
	{
		double result_1 = *bias;
		for(int i=0; i<100; i++)
		{
			result_1 += weight[i]*u_1[i];
			//printf("u[i]=\n",u_1[i]);
		}
		//printf("The result is %f\n",result_1);
		
	}
}

__global__ void bpLSTM( double result_0, double result_1, double *w_i, double *u_i, double *b_i, double *w_f, double *u_f, double *b_f, double *w_c, double *u_c, double *b_c, double *w_o, double *u_o, double *b_o, double *weight, double *bias, double *LSTM_results, double *at_0, double *at_1, double *it_0, double *it_1, double *ft_0, double *ft_1, double *ot_0, double *ot_1, double *statet_0, double *statet_1, double *u_0, double *u_1)
{
       int x = threadIdx.x;
       double i[2];
       i[0] = input_0; 
       i[1] = input_1;
       __shared__ double u_delta[100];
       __shared__ double O_delta[100];
       __shared__ double I_delta[100];
       __shared__ double F_delta[100];
       __shared__ double A_delta[100];
       __shared__ double state_delta[100];
       __shared__ double O_future_delta[100];
       __shared__ double I_future_delta[100];
       __shared__ double F_future_delta[100];
       __shared__ double A_future_delta[100];
       __shared__ double state_future_delta[100];
       __shared__ double forget_gate_future[100];

       __shared__ double u_pre[100];
       __shared__ double state_pre[100];
      
  
       O_future_delta[x] = 0.0;
       I_future_delta[x] = 0.0;
       F_future_delta[x] = 0.0;
       A_future_delta[x] = 0.0;
       state_future_delta[x] = 0.0;
       forget_gate_future[x] = 0.0;
       u_pre[x] = 0.0;
       state_pre[x]= 0.0;
     
       __syncthreads();
       u_delta[x] = 0;
       weight[x] += alpha * (input_1 - result_1) * dsigmoid(result_1) * u_1[x];
       u_delta[x] += (input_1 - result_1) * dsigmoid(result_1) * weight[x];
       for (int k = 0; k < 100; k++)
       {
    	    u_delta[x] += I_future_delta[k] * u_i[x*100+k];
    	    u_delta[x] += F_future_delta[k] * u_f[x*100+k];
    	    u_delta[x] += O_future_delta[k] * u_o[x*100+k];
    	    u_delta[x] += A_future_delta[k] * u_c[x*100+k];
       }
       O_delta[x] = u_delta[x] * tanh(statet_1[x]) * dsigmoid(ot_1[x]);
       state_delta[x] = u_delta[x] * ot_1[x] * dtanh(statet_1[x]) + state_future_delta[x] *  forget_gate_future[x];
       F_delta[x] = state_delta[x] * statet_0[x] * dsigmoid(ft_1[x]);
       I_delta[x] = state_delta[x] * at_1[x] * dsigmoid(it_1[x]);
       A_delta[x] = state_delta[x] * it_1[x] * dtanh(at_1[x]);


       for(int k = 0; k < 100; k++)
       {
    	    u_i[k*100+x] += alpha * I_delta[x] * u_0[k];
    	    u_f[k*100+x] += alpha * F_delta[x] * u_0[k];
    	    u_o[k*100+x] += alpha * O_delta[x] * u_0[k];
    	    u_c[k*100+x] += alpha * A_delta[x] * u_0[k];
       }

       w_i[x] += alpha * I_delta[x] * i[1];
       w_f[x] += alpha * F_delta[x] * i[1];
       w_o[x] += alpha * O_delta[x] * i[1];
       w_c[x] += alpha * A_delta[x] * i[1];

       __syncthreads();
    
      // double *u_pre = new double[100];
       //double *state_pre = new double[100];
      // for (int j = 0; j < 100; j++) //第一个隐藏层输出，相当于h_(t-1)
      // {
       //     u_pre[j] = 0.0;
	//    state_pre[j] = 0.0;
      // }
       u_delta[x] = 0;
       weight[x] += alpha * (input_0 - result_0) * dsigmoid(result_0) * u_0[x];
       u_delta[x] += (input_0 - result_0) * dsigmoid(result_0) * weight[x];
       for (int  k = 0; k < 100; k++)
       {
    	    u_delta[x] += I_delta[k] * u_i[x*100+k];
    	    u_delta[x] += F_delta[k] * u_f[x*100+k];
    	    u_delta[x] += O_delta[k] * u_o[x*100+k];
    	    u_delta[x] += A_delta[k] * u_c[x*100+k];
       }
       O_delta[x] = u_delta[x] * tanh(statet_0[x]) * dsigmoid(ot_0[x]);
       state_delta[x] = u_delta[x] * ot_0[x] * dtanh(statet_0[x]) +  statet_1[x] * ft_1[x];
       F_delta[x] = u_delta[x] * state_pre[x] * dsigmoid(ft_0[x]);
       I_delta[x] = u_delta[x] * at_0[x] * dsigmoid(it_0[x]);
       A_delta[x] = u_delta[x] * it_0[x] * dtanh(at_0[x]);
       for (int k = 0; k < 100; k++)
       {
    	    u_i[k*100+x] += alpha * I_delta[x] * u_pre[k];
    	    u_f[k*100+x] += alpha * F_delta[x] * u_pre[k];
    	    u_o[k*100+x] += alpha * O_delta[x] * u_pre[k];
    	    u_c[k*100+x] += alpha * A_delta[x] * u_pre[k];
       }
       w_i[x] += alpha * I_delta[x] * i[0];
       w_f[x] += alpha * F_delta[x] * i[0];
       w_o[x] += alpha * O_delta[x] * i[0];
       w_c[x] += alpha * A_delta[x] * i[0];
       __syncthreads();

}




void NeuralNetwork()
{
	cudaError_t err;
	//cudaEvent_t start, stop;
	double *w_i = (double*) malloc (100 * NUM * sizeof(double));
	double *u_i = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_i = (double*) malloc (100 * NUM * sizeof(double));
	double *w_f = (double*) malloc (100 * NUM * sizeof(double));
	double *u_f = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_f = (double*) malloc (100 * NUM * sizeof(double));
	double *w_c = (double*) malloc (100 * NUM * sizeof(double));
	double *u_c = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_c = (double*) malloc (100 * NUM * sizeof(double));
	double *w_o = (double*) malloc (100 * NUM * sizeof(double));
	double *u_o = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_o = (double*) malloc (100 * NUM * sizeof(double));
	double *weights = (double*) malloc (101 * NUM * sizeof(double));
	double *bias = (double*) malloc (sizeof(double));
        double *at_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *it_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *ft_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *ot_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *statet_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *u_0 = (double*) malloc (100 * NUM * sizeof(double));
        double *at_1 = (double*) malloc (100 * NUM * sizeof(double));
        double *it_1 = (double*) malloc (100 * NUM * sizeof(double));
        double *ft_1 = (double*) malloc (100 * NUM * sizeof(double));
        double *ot_1 = (double*) malloc (100 * NUM * sizeof(double));
        double *statet_1 = (double*) malloc (100 * NUM * sizeof(double));
        double *u_1 = (double*) malloc (100 * NUM * sizeof(double));


    




	InitHostMem(w_i, u_i, b_i, w_f, u_f, b_f, w_c, u_c, b_c, w_o, u_o, b_o, weights, bias);
	//cout<<*bias<<endl;
	double *w_i_device;
	err = cudaMalloc((void**) &w_i_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_i_device;
	err = cudaMalloc((void**) &u_i_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_i_device;
	err = cudaMalloc((void**) &b_i_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_f_device;
	err = cudaMalloc((void**) &w_f_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_f_device;
	err = cudaMalloc((void**) &u_f_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_f_device;
	err = cudaMalloc((void**) &b_f_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_c_device;
	err = cudaMalloc((void**) &w_c_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_c_device;
	err = cudaMalloc((void**) &u_c_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_c_device;
	err = cudaMalloc((void**) &b_c_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_o_device;
	err = cudaMalloc((void**) &w_o_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_o_device;
	err = cudaMalloc((void**) &u_o_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_o_device;
	err = cudaMalloc((void**) &b_o_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *weights_device;
	err = cudaMalloc((void**) &weights_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *bias_device;
	err = cudaMalloc((void**) &bias_device, 1* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *LSTM_results;
	err = cudaMalloc((void**) &LSTM_results, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Malloc completed\n");
	//Start Memory Copy
	err = cudaMemcpy(w_i_device, w_i, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 1(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_i_device, u_i, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 2(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_i_device, b_i, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 3(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_f_device, w_f, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 4(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_f_device, u_f, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 5(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_f_device, b_f, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 6(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_c_device, w_c, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 7(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_c_device, u_c, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 8(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_c_device, b_c, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 9(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_o_device, w_o, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_o_device, u_o, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_o_device, b_o, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(weights_device, weights, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(bias_device, bias, sizeof(double)*1*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device bias data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
    



        double *at_0_device;
	err = cudaMalloc((void**) &at_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *it_0_device;
	err = cudaMalloc((void**) &it_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *ft_0_device;
	err = cudaMalloc((void**) &ft_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);

        }
        double *ot_0_device;
	err = cudaMalloc((void**) &ot_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *statet_0_device;
	err = cudaMalloc((void**) &statet_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *u_0_device;
	err = cudaMalloc((void**) &u_0_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
    





        double *at_1_device;
	err = cudaMalloc((void**) &at_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *it_1_device;
	err = cudaMalloc((void**) &it_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *ft_1_device;
	err = cudaMalloc((void**) &ft_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);

        }
        double *ot_1_device;
	err = cudaMalloc((void**) &ot_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *statet_1_device;
	err = cudaMalloc((void**) &statet_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        double *u_1_device;
	err = cudaMalloc((void**) &u_1_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

    //printf("Malloc completed\n");
	//Start Memory Copy
	err = cudaMemcpy(at_0_device, at_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(it_0_device, it_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(ft_0_device, ft_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(ot_0_device, ot_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(statet_0_device, statet_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(u_0_device, u_0, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(at_1_device, at_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(it_1_device, it_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(ft_1_device, ft_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(ot_1_device, ot_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(statet_1_device, statet_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
        err = cudaMemcpy(u_1_device, u_1, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }

    
//double at, it, ft, ot, statet, output_state
	//printf("Memcpy completed\n");
	dim3 n_threads(100,1,1);
	dim3 n_blocks(1,1,1);
	double result_0 = 0;
	double result_1 = 0;

	ExecuteLSTM<<<n_blocks,n_threads>>>(result_0, result_1, w_i_device, u_i_device, b_i_device, w_f_device, u_f_device, b_f_device, w_c_device, u_c_device, b_c_device, w_o_device, u_o_device, b_o_device, weights_device, bias_device, LSTM_results, at_0_device, at_1_device, it_0_device, it_1_device, ft_0_device, ft_1_device, ot_0_device, ot_1_device, statet_0_device, statet_1_device, u_0_device, u_1_device);
        cudaDeviceSynchronize();
        printf("finishing executing forward propagation"); 
        //cudaThreadSynchronize();
        bpLSTM<<<n_blocks,n_threads>>>(result_0, result_1, w_i_device, u_i_device, b_i_device, w_f_device, u_f_device, b_f_device, w_c_device, u_c_device, b_c_device, w_o_device, u_o_device, b_o_device, weights_device, bias_device, LSTM_results, at_0_device, at_1_device, it_0_device, it_1_device, ft_0_device, ft_1_device, ot_0_device, ot_1_device, statet_0_device, statet_1_device, u_0_device, u_1_device);
        cudaDeviceSynchronize();
        //cudaThreadSynchronize();
        printf("finishing executing back propagation");
}
