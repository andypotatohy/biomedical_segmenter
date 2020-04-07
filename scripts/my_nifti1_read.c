/*********************************************************************
 * Adapted from nifti1_read_write.c by Kate Fissell,
 * University of Pittsburgh, May 2005.
 * Yu (Andy) Huang, MSK-CCNY AI Partnership, March 2020
 *********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nifti1.h"

//typedef float MY_DATATYPE;

#define MIN_HEADER_SIZE 348
#define NII_HEADER_SIZE 352

const float *main(char *img_file) //, int xRange)
//void main(char *hdr_file, char *data_file, float *data) //, int xRange)
//void main(char *hdr_file, char *data_file, int xRange[])
//main(argc,argv)
//int argc;
//char *argv[];
{
nifti_1_header hdr;
FILE *fp;
int ret,i;
//double total;
float *data=NULL;


/********** open and read header */
fp = fopen(img_file,"r");
if (fp == NULL) {
        fprintf(stderr, "\nError opening header file %s\n",img_file);
        exit(1);
}
ret = fread(&hdr, MIN_HEADER_SIZE, 1, fp);
if (ret != 1) {
        fprintf(stderr, "\nError reading header file %s\n",img_file);
        exit(1);
}
fclose(fp);


///********** print a little header information */
//fprintf(stderr, "\n%s header information:",img_file);
//fprintf(stderr, "\nXYZT dimensions: %d %d %d %d",hdr.dim[1],hdr.dim[2],hdr.dim[3],hdr.dim[4]);
//fprintf(stderr, "\nDatatype code and bits/pixel: %d %d",hdr.datatype,hdr.bitpix);
//fprintf(stderr, "\nScaling slope and intercept: %.6f %.6f",hdr.scl_slope,hdr.scl_inter);
//fprintf(stderr, "\nByte offset to data in datafile: %ld",(long)(hdr.vox_offset));
//fprintf(stderr, "\n");


/********** open the datafile, jump to data offset */
fp = fopen(img_file,"r");
if (fp == NULL) {
        fprintf(stderr, "\nError opening data file %s\n",img_file);
        exit(1);
}

ret = fseek(fp, (long)(hdr.vox_offset), SEEK_SET);
if (ret != 0) {
        fprintf(stderr, "\nError doing fseek() to %ld in data file %s\n",(long)(hdr.vox_offset), img_file);
        exit(1);
}


/********** allocate buffer and read first 3D volume from data file */
data = (float *) malloc(sizeof(float) * hdr.dim[1]*hdr.dim[2]*hdr.dim[3]);
if (data == NULL) {
        fprintf(stderr, "\nError allocating data buffer for %s\n",img_file);
        exit(1);
}
ret = fread(data, sizeof(float), hdr.dim[1]*hdr.dim[2]*hdr.dim[3], fp);
if (ret != hdr.dim[1]*hdr.dim[2]*hdr.dim[3]) {
        fprintf(stderr, "\nError reading volume 1 from %s (%d)\n",img_file,ret);
        exit(1);
}
fclose(fp);


/********** scale the data buffer  */
if (hdr.scl_slope != 0) {
        for (i=0; i<hdr.dim[1]*hdr.dim[2]*hdr.dim[3]; i++)
                data[i] = (data[i] * hdr.scl_slope) + hdr.scl_inter;
}


///********** print mean of data */
//total = 0;
//for (i=0; i<hdr.dim[1]*hdr.dim[2]*hdr.dim[3]; i++)
//        total += data[i];
////total /= (hdr.dim[1]*hdr.dim[2]*hdr.dim[3]);
//fprintf(stderr, "\nMean of volume 1 in %s is %f\n",img_file,total);

//for (i=234567; i<234578; i++)
//        printf("%f, ", data[i]);
//printf("\n");

return(data);
//exit(0);
}