# Introduction #

This code-repository hosts several projects that make use of the Proximal Distance MM framework for optimization. Current algorithms include implementations for solving convex clustering and L0 regression. We will post documentation on usage on this Wiki, and any new updates, patches, etc as well.

Please click the Source link, and copy and paste the the svn command for anonymous check out to get the latest source for building. For example, you'll want to type the following in your terminal:
```
svn checkout http://proxmap-mm.googlecode.com/svn/trunk/ proxmap-mm
```
> See below for further usage details on particular projects.

# General compilation notes #

You will want to first enter the src directory and edit the **common.mk**, which lets the makefile know where your library installations are located. The following libraries are required by the framework:
  * Boost: http://www.boost.org/
  * GSL: http://www.gnu.org/software/gsl/

You'll want to set the proper fields in common.mk to the locations of the header files and the shared libraries for these packages. On my installation, the fields look like (your locations should be similar):
```
[garykche@epigraph src]$ grep ^BOOST common.mk 
BOOST_INC_FLAGS = -I/usr/include/boost
BOOST_LIB_FLAGS = -L/usr/lib64
[garykche@epigraph src]$ grep ^GSL common.mk 
GSL_LIB_FLAGS = -lgsl -lgslcblas
[garykche@epigraph src]$ 
```

If you have a GPU, please set **use\_gpu** equal to 1. You will need to make sure you have the OpenCL library installed and make sure the OpenCL samples work. Please edit the values in **common.mk** for fields suffixed with INC\_FLAGS and LIB\_FLAGS to reflect where your header files and .so shared objects are installed.

After customization of **common.mk**, you can now run **make**, and a new executable should be generated in the **bin** directory on the same level as **src**. Please contact me via the issue tracker if you run into any difficulties with compilation.

# Convex Clustering #

In the life sciences, hierarchical clustering has achieved a position of pre-eminence due to its ability to capture multiple levels of data granularity. Despite its merits, hierarchical clustering is greedy by nature and often produces spurious clusters, particularly in the presence of substantial noise. A relatively new alternative is known as convex clustering. Although convex clustering is more computationally demanding, it enjoys several advantages over hierarchical clustering and other traditional methods of clustering. Convex clustering delivers a uniquely defined clustering path that partially obviates the need for choosing an optimal number of clusters. Along the path small clusters gradually coalesce to form larger clusters. Clustering can be guided by external information through appropriately defined similarity weights. Documentation for our convex clustering can be found at ConvexClusterDocs.