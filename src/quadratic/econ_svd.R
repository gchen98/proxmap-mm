econ_svd = function(filename){
    X<-as.matrix(read.table(filename,header=F))
    m           = dim(X)[1]
    n           = dim(X)[2]
    my.svd = svd(X, nu = 0, nv = min(m,n))
    s = as.matrix(my.svd$d) # singular values of X
    V = as.matrix(my.svd$v) # right singular vectors of X
    d = s^2    # eigenvalues of A
    # save output to files
    write.table(V,"right_singular_vectors.txt",quote=F,row.names=F,col.names=F,sep="\t")
    write.table(d,"eigenvalues.txt",quote=F,row.names=F,col.names=F,sep="\t")
}

