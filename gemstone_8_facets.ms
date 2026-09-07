/*  gemopt radial pattern — generated from PatternLayout
    N = 8   sector = 45.00000 deg
    stroke = 0.02000   offset = 90.00000
    Scripting menu -> Run Script.
*/
n = 8
sectorDeg = 45.00000000
offsetDeg = 90.00000000
radius = 50.00000000
stroke = 1.00000000

-- N-sided cylinder: Sides = N, rotated to the same offset as the PNG.
cyl = cylinder radius:radius height:(radius * 0.08) sides:n capstart:on capend:on
cyl.name = "gemopt_ngon"
rotate cyl (angleaxis offsetDeg [0,0,1])

fn gemSpline pts name = (
    local sp = splineShape name:name
    addNewSpline sp
    for p in pts do addKnot sp 1 #corner #line p
    close sp 1
    updateShape sp
    sp
)

gemSpline #([0.00000, 0.00000, 0.00000], [6.50562, 15.70595, 0.00000], [0.00000, 50.00000, 0.00000], [-6.50562, 15.70595, 0.00000]) "wedge_00"
gemSpline #([0.00000, 0.00000, 0.00000], [-6.50562, 15.70595, 0.00000], [-35.35534, 35.35534, 0.00000], [-15.70595, 6.50562, 0.00000]) "wedge_01"
gemSpline #([0.00000, 0.00000, 0.00000], [-15.70595, 6.50562, 0.00000], [-50.00000, 0.00000, 0.00000], [-15.70595, -6.50562, 0.00000]) "wedge_02"
gemSpline #([0.00000, 0.00000, 0.00000], [-15.70595, -6.50562, 0.00000], [-35.35534, -35.35534, 0.00000], [-6.50562, -15.70595, 0.00000]) "wedge_03"
gemSpline #([0.00000, 0.00000, 0.00000], [-6.50562, -15.70595, 0.00000], [-0.00000, -50.00000, 0.00000], [6.50562, -15.70595, 0.00000]) "wedge_04"
gemSpline #([0.00000, 0.00000, 0.00000], [6.50562, -15.70595, 0.00000], [35.35534, -35.35534, 0.00000], [15.70595, -6.50562, 0.00000]) "wedge_05"
gemSpline #([0.00000, 0.00000, 0.00000], [15.70595, -6.50562, 0.00000], [50.00000, -0.00000, 0.00000], [15.70595, 6.50562, 0.00000]) "wedge_06"
gemSpline #([0.00000, 0.00000, 0.00000], [15.70595, 6.50562, 0.00000], [35.35534, 35.35534, 0.00000], [6.50562, 15.70595, 0.00000]) "wedge_07"
gemSpline #([0.00000, 50.00000, 0.00000], [-35.35534, 35.35534, 0.00000], [-50.00000, 0.00000, 0.00000], [-35.35534, -35.35534, 0.00000], [-0.00000, -50.00000, 0.00000], [35.35534, -35.35534, 0.00000], [50.00000, -0.00000, 0.00000], [35.35534, 35.35534, 0.00000]) "ngon_outline"

c0 = circle radius:8.00000000 name:"halo_00"
c1 = circle radius:14.85714286 name:"halo_01"
c2 = circle radius:21.71428571 name:"halo_02"
c3 = circle radius:28.57142857 name:"halo_03"
c4 = circle radius:35.42857143 name:"halo_04"
c5 = circle radius:42.28571429 name:"halo_05"
c6 = circle radius:49.14285714 name:"halo_06"
c7 = circle radius:56.00000000 name:"halo_07"

print ("gemopt pattern N=" + n as string + " sector=" + sectorDeg as string)
