
Many of the API and library follows the structure of the OP2 high-level
library for unstructured mesh applications [@op2]. However the
structured mesh domain is distinct from the unstructured mesh
applications domain due to the implicit connectivity between
neighbouring mesh elements (such as vertices, cells) in structured
meshes/grids. The key idea is that operations involve looping over a
"rectangular" multi-dimensional set of grid points using one or more
"stencils" to access data. In multi-block grids, we have several
structured blocks. The connectivity between the faces of different
blocks can be quite complex, and in particular they may not be oriented
in the same way, i.e. an $i,j$ face of one block may correspond to the
$j,k$ face of another block. This is awkward and hard to handle simply.

To clarify some of the important issues in designing the API, we note
here some needs connected with a 3D application:

-   When looping over the interior with loop indices $i,j,k$, often
    there are 1D arrays which are referenced using just one of the
    indices.

-   To implement boundary conditions, we often loop over a 2D face,
    accessing both the 3D dataset and data from a 2D dataset.

-   To implement periodic boundary conditions using dummy "halo" points,
    we sometimes have to copy one plane of boundary data to another.
    e.g. if the first dimension has size $I$ then we might copy the
    plane $i=I\!-\!2$ to plane $i=0$, and plane $i=1$ to plane
    $i=I\!-\!1$.

-   In multigrid, we are working with two grids with one having twice as
    many points as the other in each direction. To handle this we
    require a stencil with a non-unit stride.

-   In multi-block grids, we have several structured blocks. The
    connectivity between the faces of different blocks can be quite
    complex, and in particular they may not be oriented in the same way,
    i.e. an $i,j$ face of one block may correspond to the $j,k$ face of
    another block. This is awkward and hard to handle simply.

The latest proposal is to handle all of these different requirements
through stencil definitions.
