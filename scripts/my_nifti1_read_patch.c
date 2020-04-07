/*********************************************************************
 * Adapted from nifti1_read_write.c by Kate Fissell,
 * University of Pittsburgh, May 2005.
 * Yu (Andy) Huang, MSK-CCNY AI Partnership, March 2020
 !!!!!!! NOTE THIS PROGRAM ASSUMES THE DATA TO BE READ IS ALWAYS FLOAT 32BIT!!!!!!!!!
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nifti1.h"

//typedef float MY_DATATYPE;

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

const float *main(char *img_file, int *low, int *high, int numOfVoxels) //, int xRange)
//void main(char *hdr_file, char *data_file, float *data) //, int xRange)
//void main(char *hdr_file, char *data_file, int xRange[])
//main(argc,argv)
//int argc;
//char *argv[];
{
nifti_1_header hdr;
FILE *fp;
int ret,i,j,k;
//double total;
float *data;
//float data[numOfVoxels];
//float *buf=NULL;


/********** open and read header */
fp = fopen(img_file,"r");
if (fp == NULL) {
        fprintf(stderr, "\nError opening header file %s\n",img_file);
        exit(1);
}
//ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
ret = fread(&hdr, NII_HEADER_SIZE, 1, fp);
if (ret != 1) {
        fprintf(stderr, "\nError reading header file %s\n",img_file);
        exit(1);
}
//fclose(fp);
rewind(fp); // header is tricky to track the pointer


///********** print a little header information */
////fprintf(stderr, "\n%s header information:",img_file);
//fprintf(stderr, "\nXYZT dimensions: %d %d %d %d",hdr.dim[1],hdr.dim[2],hdr.dim[3],hdr.dim[4]);
//fprintf(stderr, "\nDatatype code and bits/pixel: %d %d",hdr.datatype,hdr.bitpix);
//fprintf(stderr, "\nScaling slope and intercept: %.6f %.6f",hdr.scl_slope,hdr.scl_inter);
//fprintf(stderr, "\nByte offset to data in datafile: %ld",(long)(hdr.vox_offset));
//fprintf(stderr, "\n");


///* open the datafile */
//fp = fopen(img_file,"r");
//if (fp == NULL) {
//        fprintf(stderr, "\nError opening data file %s\n",img_file);
//        exit(1);
//}

//ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
//if (ret != 0) {
//        fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",(long)(hdr.vox_offset), img_file);
//        exit(1);
//}


/********** allocate buffer and read first 3D volume from data file */
//data = (float *) malloc(sizeof(float) * hdr.dim[1]*hdr.dim[2]*hdr.dim[3]);
data = (float *) malloc(sizeof(float) * numOfVoxels);
if (data == NULL) {
        fprintf(stderr, "\nError allocating data buffer for %s\n",img_file);
        exit(1);
}

int lenIndLinear;
lenIndLinear = (high[1]-low[1]+1)*(high[2]-low[2]+1)*2+2;
int indLinear[lenIndLinear];
//indLinear = (int *) malloc(sizeof(int) * (lenIndLinear));
indLinear[0]=0;
int n=1;
for (k=low[2]; k<=high[2]; k++)
    {
    for (j=low[1]; j<=high[1]; j++) // Matlab style, includes the up limit
        {
            indLinear[n] = low[0] + hdr.dim[1]*(j + hdr.dim[2]*k);
            indLinear[n+1] = high[0] + hdr.dim[1]*(j + hdr.dim[2]*k);
            n+=2;
        }
    }
//indLinear[n] = hdr.dim[1]*hdr.dim[2]*hdr.dim[3];
// below from MRFseg project (same thing):
// offset = i0 + dimStd[0]*(i1 + dimStd[1]*i2);
///* NOTE: convert Matlab indexing into linear indexing, for COLUMN-first indexing (Matlab matrix style),
//we have A(i0,i1,i2), is equivalent to A[i0 + dimStd[0]*(i1 + dimStd[1]*i2)], 0-based */

//for (i=0; i<3; i++)
//        printf("%d, ", low[i]);
//printf("\n");
//for (i=0; i<3; i++)
//        printf("%d, ", high[i]);
//printf("\n");
//printf("%d\n",numOfVoxels);

int offset, readLen, bufLen;
bufLen = high[0]-low[0]+1;
float buf[bufLen];
//buf =  (float *) malloc(sizeof(float) * bufLen);
n=0;
for (i=0; i<lenIndLinear-2; i+=2)
       {
//        printf("fseek from %d to %d\n", indLinear[i],indLinear[i+1]);
       if (i==0)
          { 
           offset = indLinear[i+1] * sizeof(float);
           ret = fseek(fp, (long)(offset)+(long)(hdr.vox_offset), SEEK_SET);
           // first jump to first linear index from origin, bypass the header
           // NOTE (long) add is different from just (int) add
          }
       else
          {
            offset = (indLinear[i+1]-indLinear[i]-1) * sizeof(float);
            ret = fseek(fp, (long)(offset), SEEK_CUR);
            // non-first: fseek from current pointer location
            // note -1 in the offset, as fread will put fp to the location after the last byte read (ie, last byte + 1)
          }
//        printf("offset %d\n",offset);
//        ret = fseek(fp, (long)(offset)+(long)(hdr.vox_offset), SEEK_SET); // jump to patch
//                        NOTE (long) add is different from just (int) add
        if (ret != 0) 
        { fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",(long)(offset), img_file);
        exit(1); }
//        printf("fread from %d to %d\n", indLinear[i+1],indLinear[i+2]);
        readLen = indLinear[i+2]-indLinear[i+1]+1;
//        printf("readLen %d\n",readLen);
        ret = fread(buf, sizeof(float), readLen, fp);
        // read to a buffer, as the pointer to buf will not automatically move after fread is done
        if (ret != readLen)
        { fprintf(stderr, "\nError reading from %s (%d)\n",img_file,ret);
          exit(1); }
//        rewind(fp);
        
        for (j=0; j<bufLen; j++)
            {
            data[n] = (buf[j] * hdr.scl_slope) + hdr.scl_inter; //scale the data 
            n++;
            }
        
       }
//printf("fseek from %d to %d\n", indLinear[i],indLinear[i+1]);
//printf("%ld\n",sizeof(float));

////ret = fread(data, sizeof(float), hdr.dim[1]*hdr.dim[2]*hdr.dim[3], fp);
////if (ret != hdr.dim[1]*hdr.dim[2]*hdr.dim[3]) {
////        fprintf(stderr, "\nError reading volume 1 from %s (%d)\n",img_file,ret);
////        exit(1);
////}
fclose(fp);
//
//
///********** scale the data buffer  */
//if (hdr.scl_slope != 0) {
//        for (i=0; i<hdr.dim[1]*hdr.dim[2]*hdr.dim[3]; i++)
//                data[i] = (data[i] * hdr.scl_slope) + hdr.scl_inter;
//}


///********** print mean of data */
//total = 0;
//for (i=0; i<hdr.dim[1]*hdr.dim[2]*hdr.dim[3]; i++)
//        total += data[i];
////total /= (hdr.dim[1]*hdr.dim[2]*hdr.dim[3]);
//fprintf(stderr, "\nMean of volume 1 in %s is %f\n",img_file,total);

//for (i=0; i<numOfVoxels; i++)
//        printf("%f, ", data[i]);
//printf("\n");

//free(buf);
//free(data);
return(data);
exit(0);
}