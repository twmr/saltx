// see https://www.firedrakeproject.org/demos/immersed_fem.py.html

maxh_cav = 0.11;
maxh_air = 0.15;
maxh_pml = 0.15;

// rectangle 12.0, 4.0, maxh = maxh_air
Point(1) = {-1.2,  1.2, 0, maxh_air};
Point(2) = {-1.2, -1.2, 0, maxh_air};
Point(3) = { 1.2, -1.2, 0, maxh_air};
Point(4) = { 1.2,  1.2, 0, maxh_air};

// circle radius 1.0, maxh = 0.1
Point(5) = { 0,  0, 0, maxh_cav};
Point(6) = { 1,  0, 0, maxh_cav};
Point(7) = {-1,  0, 0, maxh_cav};
Point(8) = { 0,  1, 0, maxh_cav};
Point(9) = { 0, -1, 0, maxh_cav};

// outer rectangle 14 x 6 maxh = maxh_pml
Point(11) = {-1.8,  1.8, 0, maxh_pml};
Point(12) = {-1.8, -1.8, 0, maxh_pml};
Point(13) = { 1.8, -1.8, 0, maxh_pml};
Point(14) = { 1.8,  1.8, 0, maxh_pml};

// outer rectangle vertical
Point(15) = {-1.2,  1.8, 0, maxh_pml};
Point(16) = {-1.2, -1.8, 0, maxh_pml};
Point(17) = { 1.2, -1.8, 0, maxh_pml};
Point(18) = { 1.2,  1.8, 0, maxh_pml};

// outer rectangle horizontal
Point(19) = {-1.8,  1.2, 0, maxh_pml};
Point(20) = {-1.8, -1.2, 0, maxh_pml};
Point(21) = { 1.8, -1.2, 0, maxh_pml};
Point(22) = { 1.8,  1.2, 0, maxh_pml};

// random point at which the mesh-maxh is set to 0.1
// Point(15) = { 6.5,  2.5, 0, 0.1};


// create the curved loops
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Circle(5) = {8, 5, 6};
Circle(6) = {6, 5, 9};
Circle(7) = {9, 5, 7};
Circle(8) = {7, 5, 8};

Line(11) = {11, 15};
Line(12) = {15, 1};
Line(13) = {1, 19};
Line(14) = {19, 11};

Line(15) = {2, 20};
Line(16) = {20, 19};
Line(17) = {2, 16};
Line(18) = {16, 12};
Line(19) = {12, 20};


Line(20) = {3, 17};
Line(21) = {17, 16};
Line(22) = {3, 21};
Line(23) = {21, 13};
Line(24) = {13, 17};


Line(25) = {4, 22};
Line(26) = {22, 21};
Line(27) = {4, 18};
Line(28) = {18, 14};
Line(29) = {14, 22};
Line(30) = {15, 18};


// now let's glue together 4 line segements
Curve Loop(31) = {11, 12, 13, 14};

Curve Loop(32) = {-13, -4, 15, 16};
Curve Loop(33) = {-15, 17, 18, 19};

Curve Loop(34) = {-3, 20, 21, -17};

Curve Loop(35) = {22, 23, 24, -20};
Curve Loop(36) = {25, 26, -22, -2};

Curve Loop(37) = {27, 28, 29, -25};
Curve Loop(38) = {30, -27, -1, -12};

// now let's glue together the 4 curve segements
Curve Loop(39) = {8, 5, 6, 7};

// now let's glue together the 4 outer boundary of the air region
Curve Loop(40) = {1, 2, 3, 4};

Curve Loop(41) = {11, 30, 28, 29, 26, 23 ,24, 21, 18, 19, 16,14};


Plane Surface(1) = {40, 39}; // Omeag \ D
Plane Surface(2) = {39};  // D
//Plane Surface(3) = {31,32,33,34,35,36,37,38}; // PML
Plane Surface(3) = {31}; // PML1
Plane Surface(4) = {32}; // PML1
Plane Surface(5) = {33}; // PML1
Plane Surface(6) = {34}; // PML1
Plane Surface(7) = {35}; // PML1
Plane Surface(8) = {36}; // PML1
Plane Surface(9) = {37}; // PML1
Plane Surface(10) = {38}; // PML1


// mesh size at the points (I guess this is not needed, since the meshsize is already specified in the Points()
// MeshSize(1) = 0.2;
// MeshSize(2) = 0.2;
// MeshSize(3) = 0.2;
// MeshSize(4) = 0.2;

// MeshSize(11) = 0.1;
// MeshSize(19) = 0.1;
// MeshSize(20) = 0.1;
// MeshSize(12) = 0.1;

// MeshSize(6) = maxh_cav;
// MeshSize(7) = maxh_cav;
// MeshSize(8) = maxh_cav;
// MeshSize(9) = maxh_cav;


//Plane Surface(4) = {16}; //random

// Finally, we group together some edges and define Physical entities.
// Firedrake uses the tags of these physical identities to distinguish
// between parts of the mesh (see the concrete example at the end of this
// page).

//Physical Curve("HorEdges", 11) = {1, 3}; // see Line(1) and Line(3)
//Physical Curve("VerEdges", 12) = {2, 4};
Physical Curve("Circle", 50) = {8, 7, 6, 5};

// aliases for the surfaces
Physical Surface("CAV", 20) = {2};
Physical Surface("AIR", 21) = {1};
Physical Surface("PML", 22) = {3,4,5,6,7,8,9,10};	

// Mesh.MeshSizeFromCurvature = 60;