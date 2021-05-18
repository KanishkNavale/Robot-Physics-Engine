body base_link {
mass:1
inertiaTensor:[4.16666666667e-06 0 0 4.16666666667e-06 0 4.16666666667e-06]
}

body finger_base_link {
mass:1
inertiaTensor:[0.0608333333333 0 0 0.0566666666667 0 0.0108333333333]
}

shape visual finger_base_link_1 (finger_base_link) {  
Q:<t(-0.17995 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_back.stl'
colorName:fingeredu_material
visual
}

shape visual finger_base_link_1 (finger_base_link) {  
Q:<t(0.0255 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_front.stl'
colorName:fingeredu_material
visual
}

shape visual finger_base_link_1 (finger_base_link) {  
Q:<t(0.0255 0.02 0.08) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_side_left.stl'
colorName:fingeredu_material
visual
}

shape visual finger_base_link_1 (finger_base_link) {  
Q:<t(0.0255 0 0.08) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_top.stl'
colorName:fingeredu_material
visual
}

shape collision finger_base_link_0 (finger_base_link) {  
color:[.8 .2 .2 .5]
Q:<t(-0.17995 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_back.stl'
contact:-2
}

shape collision finger_base_link_0 (finger_base_link) {  
color:[.8 .2 .2 .5]
Q:<t(0.0255 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_front.stl'
contact:-2
}

shape collision finger_base_link_0 (finger_base_link) {  
color:[.8 .2 .2 .5]
Q:<t(0.0255 0.02 0.08) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_side_left.stl'
contact:-2
}

shape collision finger_base_link_0 (finger_base_link) {  
color:[.8 .2 .2 .5]
Q:<t(0.0255 0 0.08) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/base_top.stl'
contact:-2
}

body finger_upper_link {
mass:0.14854
inertiaTensor:[0.00003 0.00005 0.00000 0.00041 0.00000 0.00041]
}

shape visual finger_upper_link_1 (finger_upper_link) {  
Q:<t(0.0195 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/upper_link.stl'
colorName:fingeredu_material
visual
}

shape collision finger_upper_link_0 (finger_upper_link) {  
color:[.8 .2 .2 .5]
Q:<t(0.0195 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/upper_link.stl'
contact:-2
}

body finger_middle_link {
mass:0.14854
inertiaTensor:[0.00041 0.00000 0.00000 0.00041 0.00005 0.00003]
}

shape visual finger_middle_link_1 (finger_middle_link) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/middle_link.stl'
colorName:fingeredu_material
visual
}

shape collision finger_middle_link_0 (finger_middle_link) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/middle_link.stl'
contact:-2
}

body finger_lower_link {
mass:0.03070
inertiaTensor:[0.00012 0.00000 0.00000 0.00012 0.00000 0.00000]
}

shape visual finger_lower_link_1 (finger_lower_link) {  
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/lower_link.stl'
colorName:fingeredu_material
visual
}

shape collision finger_lower_link_0 (finger_lower_link) {  
color:[.8 .2 .2 .5]
Q:<t(0 0 0) E(0 0 0)>
type:mesh
mesh:'package://model/meshes/lower_link.stl'
contact:-2
}

body finger_tip_link {
mass:0.01
inertiaTensor:[1.66666666667e-07 0 0 1.66666666667e-07 0 1.66666666667e-07]
}

joint finger_lower_to_tip_joint (finger_lower_link finger_tip_link) {  
type:rigid
A:<t(0 -0.008 -0.16)>
}

joint finger_base_to_upper_joint (finger_base_link finger_upper_link) {  
type:hingeX
axis:[-1 0 0]
A:<t(0 0 0) E(0 0 0)>
limits:[-1.57079632679 1.57079632679]
ctrl_limits:[1000 1000 1]
}

joint finger_upper_to_middle_joint (finger_upper_link finger_middle_link) {  
type:hingeX
axis:[0 1 0]
A:<t(0 -0.014 0) E(0 0 0)>
limits:[-1.57079632679 1.57079632679]
ctrl_limits:[1000 1000 1]
}

joint finger_middle_to_lower_joint (finger_middle_link finger_lower_link) {  
type:hingeX
axis:[0 1 0]
A:<t(0 -0.03745 -0.16) E(0 0 0)>
limits:[-3.14159265359 3.14159265359]
ctrl_limits:[1000 1000 1]
}

joint base_to_finger (base_link finger_base_link) {  
type:rigid
A:<t(0 0 0.283)>
}

