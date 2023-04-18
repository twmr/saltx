// see https://www.firedrakeproject.org/demos/immersed_fem.py.html

maxh_cav = 0.11;
maxh_air = 0.15;
maxh_pml = 0.15;
maxh_circle = 0.06; // 0.08 doesn't work?!?!?


Point(1) = {0,  1.2, 0, maxh_air};
Point(3) = { 1.2, 0, 0, maxh_air};
Point(4) = { 1.2,  1.2, 0, maxh_air};

Point(5) = { 0,  0, 0, maxh_cav};
Point(6) = { 1,  0, 0, maxh_circle};
Point(8) = { 0,  1, 0, maxh_circle};

Point(14) = { 1.8,  1.8, 0, maxh_pml};
Point(15) = {0,  1.8, 0, maxh_pml};
Point(18) = { 1.2,  1.8, 0, maxh_pml};

Point(21) = { 1.8, 0, 0, maxh_pml};
Point(22) = { 1.8,  1.2, 0, maxh_pml};

// random point at which the mesh-maxh is set to 0.1
// Point(15) = { 6.5,  2.5, 0, 0.1};


// create the curved loops
Line(1) = {1, 4};
Line(2) = {4, 3};
Circle(5) = {8, 5, 6};

Line(12) = {15, 1};
Line(22) = {3, 21};

Line(25) = {4, 22};
Line(26) = {22, 21};
Line(27) = {4, 18};
Line(28) = {18, 14};
Line(29) = {14, 22};
Line(30) = {15, 18};


Line(40) = {8, 1};
Line(41) = {5, 8};
Line(42) = {6, 5};
Line(43) = {3, 6};

// now let's glue together 4 line segements
Curve Loop(59) = {5, 42, 41};
Curve Loop(60) = {40, 1, 2, 43, -5};
Curve Loop(61) = {30, -27, -1, -12};
Curve Loop(62) = {27, 28, 29, -25};
Curve Loop(63) = {25, 26, -22, -2};


Plane Surface(1) = {60, 59}; // air: Omeag \ D
Plane Surface(2) = {59};  // D
Plane Surface(3) = {61}; // PML1
Plane Surface(4) = {62}; // PML1
Plane Surface(5) = {63}; // PML1


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
//Physical Curve("Circle", 50) = {8, 7, 6, 5};

// aliases for the surfaces
Physical Surface("CAV", 20) = {2};
Physical Surface("AIR", 21) = {1};
Physical Surface("PML", 22) = {3,4,5};	

// Mesh.MeshSizeFromCurvature = 60;