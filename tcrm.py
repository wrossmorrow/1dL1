
import numpy as np

def r_1dl1( x , y , xtol=1.0e-6 , rtol=None ) : 
	
	""" Solve a 1-D L1 regression problem of the form
		
			min_b || y - b x ||_1
			
		for x,y given vectors in R^N returning a solution b in R. 
		
		Search algorithm: Note that
		
			|| y - b x ||_1 = sum_{n=1]^N | y(n) - b x(n) |
							= sum_{n=1]^N | x(n) | | r(n) - b |
		
		where r = y/x (componentwise). Suppose the indices are ordered
		so that 
		
			r(1) <= r(2) <= ... <= r(N)
			
		Then if b <= r(1)
		
			|| y - b x ||_1 = sum_{n=1}^N | x(n) | ( r(n) - b )
		
		which is strictly decreasing in b; similarly, if b >= r(N)
		
			|| y - b x ||_1 = sum_{n=1}^N | x(n) | ( b - r(n) )
		
		which is strictly increasing in b; finally, for any b in (r(1),r(N)), 
			
			|| y - b x ||_1 = sum_{n=1}^{k(b)} | x(n) | ( b - r(n) )
								+ sum_{n=k(b)+1}^N | x(n) | ( r(n) - b )
		
		where 1 < k(b) < N ensures that the summands in both sums above are 
		non-negative. Then
		
			|| y - b x ||_1 = b ( sum_{n=1}^{k(b)} |x(n)| 
										- sum_{n=k(b)+1}^N |x(n)| )
								- sum_{n=1}^{k(b)} |x(n)| r(n)
								+ sum_{n=k(b)+1}^N |x(n)| r(n)
		
		Note that, for fixed k(b), the derivative of the objective is
		
			d/db [ || y - b x ||_1 ] = P(k(b)) - Q(k(b)) 
		
		where, for suitable k, 
			
			P(k) = sum_{n=1}^k |x(n)|   and   Q(k) = sum_{n=k+1}^N |x(n)|
			
		P is strictly increasing and Q strictly decreasing in k (presuming 
		no x's are zero, which we can -- and should -- pre-process away). 
		In particular, 
			
			P(k+1) = P(k) + |x(k+1)|
			Q(k+1) = Q(k) - |x(k+1)|
		
		and thus
		
			P(k+1) - Q(k+1) = P(k) - Q(k) + 2 |x(k+1)|
			
		That is, the derivatives of the piecewise linear objective are
		themselves strictly increasing in k. The solution, b, is characterized
		by either P(k) = Q(k) (and thus one of the line segments is flat) or 
		
			P(k) < Q(k)   and   P(k+1) > Q(k+1)
			
		(and b is exactly equal to r(k+1)). This means we can search as 
		follows (after sorting x to be consistent with ascending ratios): 
		
			r <- y/x
			i <- arg sort( r )
			P <- |x(i(1))| , Q <- sum_{n=2}^N |x(i(n))| , k = 1
			while P < Q : 
				P <- P + |x(i(k+1))| , Q <- Q - |x(i(k+1))| , k <- k + 1
				if k > N : break
			if P == Q : return (r(i(k))+r(i(k-1)))/2
			else : return r(i(k))
			
		
		test log: 
	"""
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# pass through calls to ensure we are working with numpy objects below
	if not isinstance( x , np.ndarray ) : return r_1dl1( np.array( x ).flatten() , y )
	if not isinstance( y , np.ndarray ) : return r_1dl1( x , np.array( y ).flatten() )
	
	# verify and assert uniform sizes
	Sx , Sy =  x.shape  ,  y.shape
	Nx , Ny = max( Sx ) , max( Sy )
	if Nx != Ny : raise ValueError( 'ldl1: x and y are not compatibly sized' )
	x.reshape((Nx,1))
	y.reshape((Ny,1))
	N = Nx
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
	# 1. preprocess: exclude x's (and y's) that are near zero. this is formally 
	# required to compute ratios, and zero x's "drop out of" the minimization problem:
	#
	#		|| y - b x ||_1 = sum_{n=1}^N | y(n) - b x(n) | 
	#						= sum_{ n : |x(n)| > 0 } | y(n) - b x(n) | 
	#								+ sum_{ n : |x(n)| == 0 } |y(n)| 
	# 
	# Practically we use a tolerance, xtol, so
	#
	#		|| y - b x ||_1 = sum_{n=1}^N | y(n) - b x(n) | 
	#						= sum_{ n : |x(n)| > t } | y(n) - b x(n) | 
	#								+ sum_{ n : |x(n)| <= t } | y(n) - b x(n) | 
	#					   <= sum_{ n : |x(n)| > t } | y(n) - b x(n) | 
	#								+ sum_{ n : |x(n)| <= t } |y(n)| 
	#								+ t b |{ |x(n)| <= t }|
	# 
	# We're going to ignore the term t b |{ |x(n)| <= t }|. Thus suppose we can 
	# differentiate the first term, as described above, then any associated 
	# stationarity condition is "off" by at most t |{ |x(n)| <= t }|. 
	# 
	# This is a reasonable first cut, but we can and should make this a bit more precise 
	# in terms of our algorithm itself. Above we write
	# 
	# 		|| y - b x ||_1 = b ( P(k(b)) - Q(k(b)+1) ) + C(b)
	# 
	# where P(k) = sum_{n=1}^k |x(n)|, Q(k) = sum_{n=k}^N |x(n)|. Thus
	# 
	# 		P(k) <= sum_{ 1 <= n <= k : |x(n)| > t } |x(n)| + t |{ 1 <= n <= k : |x(n)| <= t }|
	# 		Q(k) <= sum_{ k <= n <= N : |x(n)| > t } |x(n)| + t |{ k <= n <= N : |x(n)| <= t }|
	# 
	# and 
	# 
	# 		P(k) - Q(k+1) = P0(k) - Q0(k+1) + t ( c(t,1:k) - c(t,k+1:N) )
	# 
	# where 
	# 
	# 		P0(k) = sum_{ 1 <= n <= k : |x(n)| > t } |x(n)|
	#		Q0(k) = sum_{ k <= n <= N : |x(n)| > t } |x(n)|
	#		and c(t,i:j) = |{ i <= n <= j : |x(n)| <= t }|
	# 
	# It is trivial to see that 
	# 
	# 		| c(t,1:k) - c(t,k+1:N) | <= c(t,1:N) = | { 1 <= n <= N : |x(n)| <= t } |
	# 
	# and thus 
	# 
	# 		| P(k) - Q(k+1) | <= | P0(k) - Q0(k+1) | + t c(t,1:N)
	# 
	# If we set t to be small enough, we can basically use P0, Q0 instead of P and Q. 
	
	m = np.abs( x ) 							# magnitudes
	i = np.where( m > xtol )[0]					# find x's that are <= xtol
	if len(i) < N : 							# are there excludable values? 
		x = x[ i ]								# exclude those small x's
		Y = np.sum( np.abs( y ) )				# objective shift initialized (not efficient)
		y = y[ i ]								# exclude corresponding y's
		Y -= np.sum( np.abs( y ) )				# finish objective shift (not efficient)
		N = x.shape[0]							# reset problem size
	else : Y = 0								# otherwise, initialize zero shift
	
	# 2. and aggregate over equal ratios... not technically necessary, but
	# does, in general, reduce the size of the search space. Depending on the data, 
	# however, equal ratios may be an extremely unlikely occurance. 
	
	r = y / x									# ratios 
	i = np.argsort( r , axis=0 )				# indices for ascending sort by ratios
	if rtol is not None : 
		d = r[i][1:] - r[i][0:-1]				# sorted differences
		# if d[j] > rtol, then r[i[j+1]] > r[i[j]] and we should keep this ratio
		j = np.where( d > rtol )[0]				# find unique elements by ratios
		#  										   (if k in j, then r[i[k+1]] > r[i[k]])
		# so which indices should we keep? TBD
		raise NotImplementedError( 'sorry... needs work' )
		
	# 3. actual search code. 
	
	# R = np.cumsum( m ) # if we do this, we can be flexible about our search... say bisection, 
	# quadratic modeling, etc. Some of these will have better scalability propoerties
	
	# p , q = 0 , N-1 # R[p] = m[0] , R[q] = sum(m)
	
	P , Q , k = -1 , np.sum( m ) , -1			# initialize search loop
	try :
		while P < Q : 							# while in-segment derivative is zero
			k += 1								# increment segment index
			if k >= N : raise IndexError()		# break out if ALL segment derivatives are zero
			P += m[i[k]]						# modify first part of derivative
			Q -= m[i[k]]						# modify second part of derivative
	except IndexError : b = r[i[N-1]]			# solution at the end
	else : 
		k -= 1									# decrement index (due to loop strategy above)
		if P == Q : b = (r[i[k]]+r[i[k-1]])/2.0 # midpoint coefficient if in-segment derivative is flat
		else : b = r[i[k]]						# "kink" coefficient if in-segment derivative changes signs
		
	# 4. finish 
	
	return b
	
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if __name__ == "__main__" : 
	
	N = 100
	a = 1.0
	b = 0.0
	s = 1.0
	
	x = 2.0 * np.random.rand( N ) - 1.0
	y = a * x + b + s * np.random.normal( N )
	
	B = r_1dl1( x , y )
	
	print( B )
	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
	