/////////////////////////////////////////
// LJ fluid GMC simulation
// CPU for wramup and GPU for parrallel computing.
// The idea is to give one thread a system for sampling
// Since it is hardcore + LJ tail, each system can only fit 
// [L] particles maxinum 
// 
// The array exist means particle in the system or not
// insertion = 1,deletion=0
// cum_exist record the exsit particle order
//
// The algrithm is same as the jupyter notebook 
// in /LJ_prototype.../LJ...ipynb
////////////////////////////////////////

#include<stdio.h>
#include<cstdlib>
#include<cmath>
#include<iostream>
#include<fstream>
#include <numeric> 

#include <curand.h>
#include <curand_kernel.h>

#include<thrust/device_vector.h>
#include<thrust/reduce.h>
#include<thrust/extrema.h>

#include <limits>
#include <cfloat>
#include <ctime>



#define  D2H cudaMemcpyDeviceToHost
#define  H2D cudaMemcpyHostToDevice
#define  D2D cudaMemcpyDeviceToDevice
#define  H2H cudaMemcpyHostToHost

using namespace std;

dim3 threads (128);
dim3 blocks  (128);

int avg_timesh = pow(2,26)/threads.x/blocks.x;
int avg_stepsh = pow(2,10);

__constant__ int avg_times; //sampling times for each thread 
__constant__ int avg_steps; //sampling decorrelate
__constant__ int intL; //int(L)
__constant__ float L ,dx, z; //z=exp(mu)
__constant__ float Va,Vb,Vc; //Vext parameters
__constant__ float epsilon; 

float energy=0;
float zh=0;
const float Lh = 32.0;
const int intLh = int(Lh);
const float dxh = 1.0/8.0;
const int Nh = Lh/dxh;
__constant__ int N;

float densityh;

float Vah,Vbh,Vch;
float epsilonh;
int warm_up_step = pow(10,6);

void ini_parameter(int seed); //initial parameter
void initial_particle(float* particles,int* existh,int seed);// initial particle coordinate
void MC_cpu_warmup(float* particlesh,int* existh,int steps,int seed);
void MC_gpu(float* particles,int* exist,int* density,float* result);
float cal_Vext_cpu (float x);
void display (float*rho,int i);//save rho and Vext

int main (void)
{   
    cudaDeviceReset();
    cudaSetDevice(1);
    int batch_size = pow(2,0); //total samples
    ofstream fout("MC_parameter.dat");
    ofstream fout2("MC_inform.dat");
    fout<<Lh<<'\t'<<dxh<<'\t'<<batch_size<<endl;
    for(int bat=0;bat<batch_size;bat++){
        ini_parameter(bat+1);
        fout2<<bat<<'\t'<<epsilonh<<'\t'<<zh<<endl;
        float *particlesh = new float [intLh];
        int *existh = new int [intLh];
        float *particles,*avg_density_result;
        int *avg_density,*exist;
        cudaMalloc((void**)&particles,sizeof(float)*threads.x*blocks.x*intLh); //2D array in 1D form [total threads][intL]
        cudaMalloc((void**)&exist,sizeof(int)*threads.x*blocks.x*intLh);
        cudaMalloc((void**)&avg_density,sizeof(int)*threads.x*blocks.x*Nh);
        cudaMalloc((void**)&avg_density_result,sizeof(float)*Nh);
        cudaMemset(particles,0,sizeof(float)*threads.x*blocks.x*intLh);
        cudaMemset(exist,0,sizeof(int)*threads.x*blocks.x*intLh);
        cudaMemset(avg_density,0,sizeof(int)*threads.x*blocks.x*Nh);
        cudaMemset(avg_density_result,0,sizeof(float)*Nh);

        initial_particle(particlesh,existh,bat+1);
        MC_cpu_warmup(particlesh,existh,warm_up_step,bat+1);

        for(int i =0;i<threads.x*blocks.x;i++)
        {
            MC_cpu_warmup(particlesh,existh,512,i*batch_size+bat);
            cudaMemcpy(&particles[i*intLh],particlesh,sizeof(float)*intLh,H2D);
            cudaMemcpy(&exist[i*intLh],existh,sizeof(float)*intLh,H2D);
        }

        cout<<"initial done"<<endl;
        cout<<"total samples"<<'\t'<<avg_timesh*blocks.x*threads.x<<endl; 
        MC_gpu(particles,exist,avg_density,avg_density_result);
        cout<<"gpu done"<<endl;
        float *resulth= new float [Nh];
        cudaMemcpy(resulth,avg_density_result,sizeof(float)*Nh,D2H);
        display(resulth,bat);
        free(particlesh);
        free(resulth);
        free(existh);
        cudaFree(exist);
        cudaFree(particles);
        cudaFree(avg_density);
        cudaFree(avg_density_result);

    }
    return 0;
}


void display (float*rho,int bat)
{
    char* s = new char[100];
    sprintf(s,"rho_%d.dat",bat);
    char* s2 = new char[100];
    sprintf(s2,"Vext_%d.dat",bat);
    ofstream fout (s);
    ofstream fout_V (s2);
    for(int i=0;i<Nh;i++){
        fout<<rho[i]/threads.x/blocks.x/avg_timesh<<endl;
        fout_V<<cal_Vext_cpu(i*dxh)<<endl;
    }
    free(s);
    free(s2);
}


void ini_parameter(int seed)
{
    cudaMemcpyToSymbol(avg_times, &avg_timesh, sizeof(avg_timesh));
    cudaMemcpyToSymbol(avg_steps, &avg_stepsh, sizeof(avg_stepsh));
    
    cudaMemcpyToSymbol(N, &Nh, sizeof(Nh));
    cudaMemcpyToSymbol(dx, &dxh, sizeof(dxh));
    cudaMemcpyToSymbol(L, &Lh, sizeof(Lh));
    cout<<"seed"<<'\t'<<seed<<endl; 
    srand(seed); 
    //cout<<"test"<<'\t'<<rand()<<endl; 
    
    Vah = (float)rand()/(float)(RAND_MAX/2)+1; //1-3
    Vbh = (float)rand()/(float)(RAND_MAX)*(Lh/2-2-Lh/4)+Lh/4; 
    Vch = (float)rand()/(float)(RAND_MAX/2)+2;//2-4

    epsilonh =  0;//(float)rand()/(float)(RAND_MAX)*1+0.5; //0.5-1.5
    zh =  (float)rand()/(float)(RAND_MAX)*1.5+1.5; //1.5-3 exp(0)=1

    /**/
    //zh = 3;
    //epsilonh=1.5;
    //Vah = 1;
    //Vbh = 8;
    //Vch = 3;
    /**/
    
    cout<<Vah<<'\t'<<Vbh<<'\t'<<Vch<<endl;

    cudaMemcpyToSymbol(epsilon, &epsilonh, sizeof(Lh));
    cudaMemcpyToSymbol(z, &zh, sizeof(zh));
    cudaMemcpyToSymbol(intL, &intLh, sizeof(intLh));

    
    cudaMemcpyToSymbol(Va, &Vah, sizeof(Vah));
    cudaMemcpyToSymbol(Vb, &Vbh, sizeof(Vbh));
    cudaMemcpyToSymbol(Vc, &Vch, sizeof(Vch));
    
}

void initial_particle(float* particles,int* exist,int seed)
{
    srand(seed);
    for(int i=0;i<intLh;i++){
        float x = (float)rand()/(float)(RAND_MAX)*Lh;
        particles[i]=x;
        x = (float)rand()/(float)(RAND_MAX)-0.5;
        exist[i]=0;
        //cout<<i<<'\t'<<particles[i]<<endl;
    }


}

__device__ __host__ float Uij(float dissq)
{
    return 4*(powf(1.0/dissq,6)-powf(1.0/dissq,3));
}


__device__ float cal_Vext_gpu (float x)
{
    float V = 0;

    if(x<L/2-Vb){
        V = Va*powf(fabs(x+Vb-L/2),Vc);
    }
    else if(x>L/2+Vb){
        V = Va*powf(fabs(x-Vb-L/2),Vc); 
    }
    else V=0;

    if(V>100)V=100;

    return V;
}

//Return distance**2
__device__ float dissq_gpu(float x1,float x2)
{
    float Dx=fabsf(x1-x2);
    if(Dx>L/2)Dx-=L;
    return Dx*Dx;
}


__device__ float particle_energy_gpu(float this_particle ,int* exist,float* particles,int p)
{
    float energy=0;
    for(int j=0;j<intL;j++){
        if(p!=j&&exist[j]==1){
            float dissq = dissq_gpu(this_particle,particles[j]);
            if(dissq<=1)return 100000000;
            energy+=Uij(dissq)*epsilon;
        }
    }
    
    energy+=cal_Vext_gpu(this_particle);

    return energy;
}

//sum
__device__ int sum_tot(int* a)
{
    int b=0;
    for(int i=0;i<intL;i++)b+=a[i];
    return b;
}

//accumilate sum
__device__ void cum_sum(int* a,int *b)
{ 
    for(int i=0;i<intL;i++)a[i]=b[i];
    for(int i=1;i<intL;i++)a[i]+=a[i-1];
}


__device__ void deletion_gpu(float* particle,int* exist,int*cum_exist,int MM,int Np,float dice)
{
        //int Np = sum_tot(exist);//current particle number
        //cum_sum(cum_exist,exist);
        //int M=curand(&state)%(Np)+1;//choose Mth exist particle (neccesary)
        for(int i=0;i<intL;i++)
        {
            if(exist[i]==1 && cum_exist[i]==MM)
            //if(exist[i]==1)
            {
                float ene = -particle_energy_gpu(particle[i],exist,particle,i);
                float prop = 1.0/L*Np/z*expf(-ene);
                if(prop>1)exist[i]=0;
                else{
                    //float dice = (curand_uniform(&state));
                    if(dice<prop)exist[i]=0;
                }
                break;
            }
        }
}

__device__ void insertion_gpu(float* particle,int* exist,int*cum_exist,float varL,int Np,float dice)
{
        //int Np = sum_tot(exist);
        for(int i=0;i<intL;i++)
        {
            if(exist[i]==0)
            {
                particle[i]=varL;
                //printf("particle[i]=%f\n",particle[i]);
                float ene = particle_energy_gpu(particle[i],exist,particle,i);
                float prop = z*L/(Np+1)*expf(-ene);
                if(prop>1)exist[i]=1;
                else{
                    //float dice = (curand_uniform(&state));
                    if(dice<prop)exist[i]=1;
                }
                break;
            }
        }
}

// each thread hold one sub system
__global__ void MC_gpu_kernel(float* particles,int* exist,int*cum_exist,int* avg_density)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=gridDim.x*blockDim.x){
        printf("out!\n");
        return;
    }
    
    curandState_t state;
    curand_init(idx,idx*idx,idx*idx*idx,&state);
    for(int iter=0;iter<avg_times;iter++){
        for(int i=0;i<avg_steps;i++){
            float prop = (curand_uniform(&state)-0.5);
            int Np = sum_tot(&exist[idx*intL]);//current particle number
            int M=idx*intL;
            float dice = (curand_uniform(&state));
            if(prop>0 && Np>0 ){
                cum_sum(&cum_exist[M],&exist[M]);
                int MM=curand(&state)%(Np)+1;//choose Mth exist particle (neccesary)
                deletion_gpu(&particles[M],&exist[M],&cum_exist[M],MM,Np,dice);
            }
            if(prop<0 && Np<intL){
                float varL =(curand_uniform(&state))*L;//new coordinate for insertion particle
                insertion_gpu(&particles[M],&exist[M],&cum_exist[M],varL,Np,dice);
            }
        }

        for(int i=0;i<intL;i++){
            int position = __float2int_rn(particles[idx*intL+i]/dx);//histogram
            //printf("position=%d,x/dr=%f",position,particles[idx*N_particles+i]/dx);
            if(exist[idx*intL+i]==1)avg_density[idx*N+position]+=1;
        }
    }
    //printf("idx=%d,avg_density=%d\n",idx,avg_density[idx*N+N/2]);
}


//sum over column to first raw
__global__ void sum_density(int* density,float* result,int num_systems)
{
    int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx>=N)return;
    for(int j=1;j<num_systems;j++)
    {
        density[idx]+=density[j*N+idx];
    }
        //printf("i=%d,density=%d\n",i,density[i]);
    result[idx]= density[idx]/dx;
}

void MC_gpu(float* particles,int* exist,int* avg_density,float* result)
{
    int* cum_exist;
    cudaMalloc((void**)&cum_exist,sizeof(int)*threads.x*blocks.x*intLh);
    MC_gpu_kernel<<<blocks,threads>>>(particles,exist,cum_exist,avg_density);
    cudaDeviceSynchronize();  
    cout<<"gpu done"<<endl;
    sum_density<<<(Nh+threads.x-1)/threads.x,threads>>>(avg_density,result,blocks.x*threads.x);
    cudaDeviceSynchronize();  
    cout<<"avg done"<<endl;
    cudaFree(cum_exist);
    
}


float cal_Vext_cpu (float x)
{
    float V = 0;
    
    if(x<Lh/2-Vbh){
        V = Vah*pow(fabs(x+Vbh-Lh/2),Vch);
    }
    else if(x>Lh/2+Vbh){
        V = Vah*pow(fabs(x-Vbh-Lh/2),Vch); 
    }
    else V=0;

    if(V>100)V=100;
    
    return V;
}

float dissq_cpu(float x1,float x2)
{
    float Dx=fabs(x1-x2);
    if(Dx>Lh/2)Dx-=Lh;
    return Dx*Dx;
}


float energy_cpu(float* particles,int *exist)
{
    float energy=0;
    
    for(int i=0;i<intLh;i++){
        for(int j=0;j<intLh;j++){
            if(i!=j && exist[i]*exist[j]==1){
                float dissq = dissq_cpu(particles[i],particles[j]);
                energy+=Uij(dissq)/2*epsilonh;
                if(dissq<1)cout<<"overlapped!!!"<<endl;
            }
        }
    }
    
    for(int i=0;i<intLh;i++){
        if(exist[i]==1)energy+=cal_Vext_cpu(particles[i]);
    }
    return energy;
}

float particle_energy_cpu(float this_particle ,int* exist,float* particles,int p)
{
    float energy=0;
    
    for(int j=0;j<intLh;j++){
        if(p!=j && exist[j]==1 ){
            float dissq = dissq_cpu(this_particle,particles[j]);
            if(dissq<1) return pow(10,8);
            energy+=Uij(dissq)*epsilonh;
        }
    }
    
    energy+=cal_Vext_cpu(this_particle);

    return energy;
}


void insertionh(float* particles,int* exist)
{
    int N = accumulate(exist,exist+intLh,0);      
    for(int i=0;i<intLh;i++)
    {
        if(exist[i]==0)
        {
            particles[i]=(float)rand()/(float)(RAND_MAX)*Lh;
            double ene = particle_energy_cpu(particles[i] ,exist,particles,i);
            double prop = zh*Lh/(N+1)*exp(-ene);
            if(prop>=1)exist[i]=1;
            else{
                float dice = (float)rand()/(float)(RAND_MAX);
                if(dice<prop)exist[i]=1;
            }
            break;
        }
    }
}

void deletionh(float* particles,int* exist)
{
    int *cum_exist = new int[intLh];
    partial_sum (exist, exist+intLh, cum_exist);
    int N = accumulate(exist,exist+intLh,0);
    int M = (int)rand()%(N)+1;

    for(int i=0;i<intLh;i++)
    {
        if(exist[i]==1 && cum_exist[i] == M)
        {
            double ene = -particle_energy_cpu(particles[i] ,exist,particles,i);
            double prop = 1.0*N/Lh/zh*exp(-ene);
            if(prop>=1)exist[i]=0;
            else{
                float dice = (float)rand()/(float)(RAND_MAX);
                if(dice<prop){
                    exist[i]=0;
                }
            }
            break;
        }
    }
    free(cum_exist);
}



void MC_cpu_warmup(float* particles,int* exist,int steps,int seed)
{
    //ofstream fout ("warmup.dat");
    srand(seed+1);
    //cout<<energy<<endl;
    int sum_N=0;
    int count=0;
    for(int i =0;i<steps;i++){
        srand(i);
        float prop = (float)rand()/(float)(RAND_MAX)-0.5;
        int N =accumulate(exist,exist+intLh,0);      
        if(prop>0 && N<intLh)insertionh(particles,exist);
        else if(prop<0 && N>0)deletionh(particles,exist);
        N =accumulate(exist,exist+intLh,0); 
        energy = energy_cpu(particles,exist);
        if(i>1000){
            sum_N+=N;
            count++;
        }
        //cout<<i<<'\t'<<N<<'\t'<<energy<<endl;
    }
    //cout<<sum_N/count/Lh<<endl;
}


