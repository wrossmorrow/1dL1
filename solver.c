
/* 
 * Solve a 1-D L1 regression problem like
 *
 * 		min_\beta || \beta x - y ||_1
 * 
 * given vectors x and y. 
 * 
 * recommended use: 
 * 
 *	> gcc solver.c -o solver
 *	> ./solver 
 * 
 * or, for a specific size, 
 * 
 *	> ./solver 1000
 * 
 * or any other positive integer or sequence of positive integers. 
 * 
 * Authored by W. Ross Morrow. Software has no warranty, and do not use for
 * commerical purposes without express written permission. 
 * 
 */
 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>


double solve( uint64_t N , double * x , double * y );
double search( uint64_t N , double * x , double * y );
double drand();
void run( uint64_t N );

int main( int argc , char * argv[] )
{
	int i;
	uint64_t N;
	
	srand( time(NULL) );
	
	if( argc > 1 ) { 
		for( i = 1 ; i < argc ; i++ ) {
			N = (uint64_t)strtol( argv[i] , NULL , 10 );
			if( N > 0 ) { run( N ); }
		}
	} else { run( 100 ); }
	
	return EXIT_SUCCESS;
}

void run( uint64_t N )
{
	uint64_t n;
	double * x , * y , b;
			
	x = ( double * ) calloc ( N , sizeof(double) );
	y = ( double * ) calloc ( N , sizeof(double) );
	if( x == NULL ) { printf( "allocation failure.\n" ); fflush(stdout); exit( EXIT_FAILURE ); }
	if( y == NULL ) { printf( "allocation failure.\n" ); fflush(stdout); exit( EXIT_FAILURE ); }
	
	for( n = 0 ; n < N ; n++ ) { x[n] = drand(); y[n] = drand(); }
	
	b =  solve( N , x , y ); printf( "solution: %0.16f\n" , b );
	b = search( N , x , y ); printf( "solution: %0.16f\n" , b );
	
	free( x );
	free( y );
}

double l1norm( uint64_t N , double b , double * x , double * y )
{
	uint64_t n;
	double r;
	
	r = 0.0;
	for( n = 0 ; n < N ; n++ ) { r += fabs( b * x[n] + y[n] ); }
	return r;
}

// uniformly distributed random number between -1 and 1
double drand()
{
	return ( 2.0 * ( (double)rand() ) / ( (double)RAND_MAX ) - 1.0 );
}

typedef struct {
	double a;
	double r;
} element;

static element * e1 , * e2;

int qsort_compare_elements( const void * p1 , const void * p2 ) 
{
	e1 = ( element * ) p1; e2 = ( element * ) p2;
	if( e1->r <= e2->r ) { return -1; } else { return 1; }
}


double solve( uint64_t N , double * x , double * y ) 
{
	// declarations
	double D , R , b;
	element * e;
	uint64_t n;
	
	if( N == 0 || x == NULL || y == NULL ) { printf( "invalid arguments to solve.\n" ); return 0.0; }
	
	// allocations
	e = ( element * ) calloc ( N , sizeof( element ) );
	if( e == NULL ) { printf( "allocation failure.\n" ); fflush(stdout); exit( EXIT_FAILURE ); }

	// preprocessing: no zero x's, sorting, aggregating x's for equal ratios
	
	for( n = 0 ; n < N ; n++ ) {
		e[n].r = y[n] / x[n];
		e[n].a = fabs( x[n] );
	}
	
	qsort( e , N , sizeof(element) , qsort_compare_elements );
	
	// algorithm
	D = e[0].a; for( n = 1 ; n < N ; n++ ) { D -= e[n].a; }
	n = 0;
	while( n < N-1 ) {
		if( D == 0.0 ) { b = ( e[n].r + e[n+1].r ) / 2.0; break; }
		if( 0.0 < D && D <= 2.0 * e[n].a ) { b = e[n].r; break; }
		n++; D += 2.0 * e[n].a;
	}
	if( n == N-1 ) { b = e[N-1].r; }
	
	// clean up
	free( e );
	
	return b;

}

double search( uint64_t N , double * x , double * y )
{
	double P , Q , b;
	element * e;
	uint64_t n;
	
	if( N == 0 || x == NULL || y == NULL ) { printf( "invalid arguments to solve.\n" ); return 0.0; }
	
	// allocations
	e = ( element * ) calloc ( N , sizeof( element ) );
	if( e == NULL ) { printf( "allocation failure.\n" ); fflush(stdout); exit( EXIT_FAILURE ); }

	// preprocessing: no zero x's, sorting, aggregating x's for equal ratios
	
	for( n = 0 ; n < N ; n++ ) {
		e[n].r = y[n] / x[n];
		e[n].a = fabs( x[n] );
	}
	
	qsort( e , N , sizeof(element) , qsort_compare_elements );
	
	// algorithm
	P = e[0].a; 
	Q = e[1].a; for( n = 2 ; n < N ; n++ ) { Q += e[n].a; }
	n = 0;
	while( n < N-1 ) {
		if( P  > Q ) { b = e[n].r; break; }
		if( P == Q ) { b = ( e[n].r + e[n+1].r ) / 2.0; break; }
		n++; P += e[n].a; Q -= e[n].a;
	}
	if( n == N-1 ) { b = e[n].r; }
	
	// clean up
	free( e );
	
	return b;
	
}
