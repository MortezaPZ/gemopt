/*  gemopt -- import an optimised stone into 3ds Max
    ------------------------------------------------------------------
    Reads a .gem file written by the Python side and rebuilds it as an
    Editable Poly with its real facets, so every facet stays selectable and
    editable exactly like the hand-built model.

    Run it from the Scripting menu, or head-less through 3dsmaxbatch with:
        3dsmaxbatch.exe import_stone.ms -mxsString gemFile:"C:\path\best.gem"

    Optional globals, all with sensible defaults:
        gemFile     full path to the .gem file (a file dialog opens if unset)
        gemName     object name in the scene            (default "gemopt_stone")
        gemMaterial "vray" | "physical" | "none"        (default "vray")
        gemIES      full path to an .ies file to create a photometric light
        gemSaveMax  where to write a .max file.  Pass "auto" to put it beside
                    the .gem with the same name.  Only 3ds Max itself can write
                    that format, so this is the only way to produce one.
        gemPattern    default picture.  The rollout can replace it.
        gemHalfWidth  half-width S of that picture, in the mesh's units.
                      Unset means the stone's own maximum plan radius.
        gemPatternFace  "pavilion" keeps the picture on the lower side.
                      Anything else maps the whole stone.
        gemoptHeadless  true skips the pattern rollout (batch / silent runs).
*/

global gemFile
global gemName
global gemMaterial
global gemIES
global gemSaveMax
global gemPattern
global gemHalfWidth
global gemPatternFace
global gemoptHeadless
global gemPatternNode
global gemPatternPath
global gemPatternBusy

fn gemDefault value fallback = (if value == undefined then fallback else value)

fn gemReadFile path = (
    local verts = #()
    local faces = #()
    local names = #()
    local ior = undefined
    local fs = openFile path mode:"r"
    if fs == undefined do throw ("cannot open " + path)

    while not eof fs do (
        local line = readLine fs
        local tokens = filterString line " \t"
        if tokens.count > 0 do (
            case tokens[1] of (
                "v": append verts [(tokens[2] as float), (tokens[3] as float), (tokens[4] as float)]
                "f": (
                    -- "f <group> i1 i2 ..."  or  "f i1 i2 ..."
                    local start = 2
                    local groupName = "facet"
                    if (tokens[2] as integer) == undefined then (
                        groupName = tokens[2]
                        start = 3
                    )
                    local idx = #()
                    for i = start to tokens.count do (
                        local n = tokens[i] as integer
                        if n != undefined do append idx n
                    )
                    if idx.count >= 3 do (
                        append faces idx
                        append names groupName
                    )
                )
                -- the refractive index the stone was optimised for, so the
                -- material matches the material rather than assuming glass
                "ior": ior = tokens[2] as float
                default: ()
            )
        )
    )
    close fs
    #(verts, faces, names, ior)
)

fn gemBuild verts faces objName = (
    -- Start from an empty Editable Poly and add the facets one by one.  Going
    -- through a triangle mesh first would weld the facets into triangles and
    -- lose exactly the structure we care about.
    -- Editable_Poly is a base-object class, not a creatable node: asking for
    -- one directly fails with "Not creatable".  An empty mesh converted to a
    -- poly is the way in, and starting it at zero verts means there is no
    -- phantom geometry to undo afterwards.
    local obj = mesh name:objName numverts:0 numfaces:0
    convertTo obj Editable_Poly
    local ep = obj.baseObject

    for v in verts do polyop.createVert ep v
    for f in faces do polyop.createPolygon ep f

    -- the empty starting vertex that Editable_Poly is born with
    if (polyop.getNumVerts ep) > verts.count do (
        local extra = #{(verts.count + 1)..(polyop.getNumVerts ep)}
        polyop.deleteVerts ep extra
    )

    update obj
    obj.pivot = [0, 0, 0]
    obj
)

/*  One material ID per facet row.

    The client's own workflow is selecting a row of facets and nudging it, and
    in the video that is done by clicking each facet in turn.  Handing the rows
    over so a whole row selects at once is the point.

    Not as named selection sets: MAXScript has no working call for creating a
    sub-object named selection set on an Editable Poly.  The only interface
    methods are namedSelCopy and namedSelPaste, which both need a live
    sub-object selection and a user interface -- unusable in a batch run, and
    the call this script used before did not exist at all, so every row was
    silently dropped.

    Material IDs do the same job and are scriptable: in Max the client picks
    Polygon level and Select By ID, and the row selects in one action.  The IDs
    carry the row names through a Multi/Sub-Object material, so they read as
    names rather than numbers.
*/
fn gemRowsToMatIDs obj names = (
    local rows = #()
    local byRow = #()
    for i = 1 to names.count do (
        local idx = findItem rows names[i]
        if idx == 0 then (
            append rows names[i]
            append byRow #{i}
        ) else (
            byRow[idx][i] = true
        )
    )
    for i = 1 to rows.count do
        polyop.setFaceMatID obj byRow[i] i
    update obj
    format "gemopt: % rows as material IDs: %\n" rows.count (rows as string)
    rows
)

fn gemGlassMaterial kind ior = (
    case kind of (
        "vray": (
            local cls = undefined
            for c in material.classes while cls == undefined do
                if (c as string) == "VRayMtl" do cls = c
            if cls != undefined then (
                local m = cls name:"gemopt_glass"
                try (
                    m.refraction_color = color 255 255 255
                    m.refraction_ior = ior
                    m.diffuse = color 8 8 8
                    m.reflection_color = color 255 255 255
                    m.reflection_fresnel = true
                    m.refraction_affectShadows = true
                ) catch ()
                format "gemopt: applied VRayMtl (IOR %)\n" ior
                m
            ) else (
                format "gemopt: V-Ray not found, falling back to Physical Material\n"
                gemGlassMaterial "physical" ior
            )
        )
        "physical": (
            local m = PhysicalMaterial name:"gemopt_glass"
            try (
                m.transparency = 1.0
                m.trans_ior = ior
                m.roughness = 0.0
                m.metalness = 0.0
            ) catch ()
            m
        )
        default: undefined
    )
)

/*  The glass, split into one named slot per row.

    Every slot is the same glass, so the render is exactly what it would have
    been with a single material -- the split exists only so the rows carry
    their names into the scene and can be picked by name instead of by number.
*/
fn gemAssignMaterial obj kind ior rows = (
    local base = gemGlassMaterial kind ior
    if base == undefined do return undefined
    if rows.count <= 1 then (
        obj.material = base
    ) else (
        local multi = Multimaterial numsubs:rows.count name:"gemopt_stone_rows"
        for i = 1 to rows.count do (
            local slot = copy base
            slot.name = "row_" + rows[i]
            multi[i] = slot
            multi.names[i] = "row_" + rows[i]
        )
        obj.material = multi
        format "gemopt: % named material slots\n" rows.count
    )
)

fn gemAddLight iesPath = (
    if iesPath == undefined or not (doesFileExist iesPath) do return undefined
    local lt = undefined
    try (
        lt = Free_Light name:"gemopt_ies"
        lt.distribution = 3                    -- photometric web
        lt.webFile = iesPath
        lt.pos = [0, 0, 60]
        lt.dir = [0, 0, -1]
        format "gemopt: created photometric light from %\n" iesPath
    ) catch (
        format "gemopt: could not create the IES light (%)\n" (getCurrentException())
    )
    lt
)

/*  Pattern on the stone.  Same two mappings as gemopt/drape.py.

    The picture is centred on the origin and covers [-S, S] in plan.
    Planar: texture (u, v) is that horizontal position.  v grows upward,
    matching a bitmap whose top is world +Y.
    Surface: in each azimuth a vertical cut walks from the crown axis to
    the culet.  Picture radius is (arc / arc_at_crown_edge) * plan radius
    of that edge, which is the first point of maximum plan radius.
    Past the edge the arc keeps growing, so the coordinate leaves the picture.

    The rollout only replaces the glass once a mode is chosen.  A headless
    run never opens it.
*/

fn gemRound8 x = (
    local scale = 1e8
    if x >= 0.0 then (floor (x * scale + 0.5)) / scale
    else (ceil (x * scale - 0.5)) / scale
)

fn gemHypot p = (
    sqrt (p.x * p.x + p.y * p.y)
)

fn gemCosRad a = cos (a * 180.0 / pi)
fn gemSinRad a = sin (a * 180.0 / pi)

fn gemAddPoint pts adj p tol = (
    -- Two cuts of one corner, or a vertex read back from a .gem, can sit a
    -- hair apart.  Rounding both to 8 decimals and then demanding an exact
    -- match splits the profile in two, so anything inside `tol` is one point.
    for i = 1 to pts.count do (
        if (distance pts[i] p) <= tol do return i
    )
    append pts p
    append adj #()
    pts.count
)

fn gemLink adj a b = (
    if (findItem adj[a] b) == 0 do append adj[a] b
    if (findItem adj[b] a) == 0 do append adj[b] a
)

fn gemClipForward p q direction = (
    local dp = dot p direction
    local dq = dot q direction
    local t = 0.0
    local axis = [0, 0, 0]
    if (dp < -1e-8) and (dq < -1e-8) do return undefined
    if (dp >= -1e-8) and (dq >= -1e-8) do return #(p, q)
    t = dp / (dp - dq)
    axis = p + t * (q - p)
    axis = [0, 0, axis.z]
    if dp >= -1e-8 then #(p, axis) else #(q, axis)
)

fn gemAddSegment pts adj p q direction tol = (
    local clipped = gemClipForward p q direction
    local a = 0
    local b = 0
    if clipped == undefined do return undefined
    a = gemAddPoint pts adj clipped[1] tol
    b = gemAddPoint pts adj clipped[2] tol
    if a != b do gemLink adj a b
    OK
)

fn gemCutTriangle corners normal pts adj direction tol = (
    local signed = #()
    local side = #()
    local hits = #()
    local unique = #()
    local s = 0.0
    local sd = 0
    local i = 0
    local j = 0
    local t = 0.0
    local seen = false
    for i = 1 to 3 do (
        s = dot corners[i] normal
        append signed s
        sd = 0
        if s > 1e-8 then sd = 1
        else if s < -1e-8 then sd = -1
        append side sd
    )
    if (side[1] == 0) and (side[2] == 0) and (side[3] == 0) do return undefined
    for i = 1 to 3 do (
        j = i + 1
        if j > 3 do j = 1
        if (side[i] == 0) and (side[j] == 0) then (
            append hits corners[i]
            append hits corners[j]
        ) else if (side[i] == 0) and (side[j] != 0) then (
            append hits corners[i]
        ) else if (side[i] != 0) and (side[j] != 0) and (side[i] != side[j]) then (
            t = signed[i] / (signed[i] - signed[j])
            append hits (corners[i] + t * (corners[j] - corners[i]))
        )
    )
    for h in hits do (
        seen = false
        for u in unique do (
            if (distance h u) <= tol do seen = true
        )
        if not seen do append unique h
    )
    if unique.count >= 2 do gemAddSegment pts adj unique[1] unique[2] direction tol
    OK
)

fn gemEdgeUsed used a b = (
    for e in used do (
        if ((e[1] == a) and (e[2] == b)) or ((e[1] == b) and (e[2] == a)) do return true
    )
    false
)

fn gemWalk start adj pts = (
    local path = #(start)
    local used = #()
    local prev = 0
    local current = start
    local guard = 0
    local nxt = #()
    local chosen = 0
    local bestR = 0.0
    local best = 0.0
    local travel = [0, 0, 0]
    local travelN = 0.0
    local step = [0, 0, 0]
    local sc = 0.0
    local r = 0.0
    while guard < 10000 do (
        guard += 1
        nxt = #()
        for other in adj[current] do (
            if (gemEdgeUsed used current other) == false do append nxt other
        )
        if nxt.count == 0 do exit
        chosen = nxt[1]
        if (nxt.count > 1) and (prev == 0) then (
            bestR = -1.0
            for n in nxt do (
                r = gemHypot pts[n]
                if r > bestR do (
                    bestR = r
                    chosen = n
                )
            )
        ) else if nxt.count > 1 then (
            travel = pts[current] - pts[prev]
            travelN = length travel
            best = -2.0
            for n in nxt do (
                step = pts[n] - pts[current]
                sc = (dot step travel) / ((length step) * travelN + 1e-15)
                if sc > best do (
                    best = sc
                    chosen = n
                )
            )
        )
        append used #(current, chosen)
        append path chosen
        prev = current
        current = chosen
    )
    path
)

-- True when the chain that starts here already meets the profile walked so far.
-- A spur off that profile must not be stitched on again at the culet.
fn gemComponentMeets path adj start = (
    local seen = #(start)
    local stack = #(start)
    local cur = 0
    local other = 0
    while stack.count > 0 do (
        cur = stack[stack.count]
        deleteItem stack stack.count
        for other in adj[cur] do (
            if (findItem path other) != 0 then (
                return true
            ) else (
                if (findItem seen other) == 0 do (
                    append seen other
                    append stack other
                )
            )
        )
    )
    false
)

fn gemProfile start adj pts = (
    local path = gemWalk start adj pts
    local guard = 0
    local ends = #()
    local i = 0
    local alive = 0
    local other = 0
    local tail = [0, 0, 0]
    local bestD = 0.0
    local nxt = 0
    local e = 0
    local d = 0.0
    local extra = #()
    local k = 0
    while guard < 1000 do (
        guard += 1
        ends = #()
        for i = 1 to pts.count do (
            if (findItem path i) == 0 and adj[i].count > 0 then (
                alive = 0
                for other in adj[i] do (
                    if (findItem path other) == 0 do alive += 1
                )
                if alive == 1 and (gemComponentMeets path adj i) == false do append ends i
            )
        )
        if ends.count == 0 do exit
        tail = pts[path[path.count]]
        bestD = 1e30
        nxt = ends[1]
        for e in ends do (
            d = distance pts[e] tail
            if d < bestD do (
                bestD = d
                nxt = e
            )
        )
        extra = gemWalk nxt adj pts
        for k in extra do append path k
    )
    path
)

fn gemSigmaAt path sigma vertex limit = (
    local bestD = 1e30
    local bestS = 0.0
    local i = 0
    local d = 0.0
    local edge = [0, 0, 0]
    local length2 = 0.0
    local t = 0.0
    local proj = [0, 0, 0]
    local dist = 0.0
    for i = 1 to path.count do (
        d = distance path[i] vertex
        if d < bestD do (
            bestD = d
            bestS = sigma[i]
        )
    )
    if bestD <= (limit * 0.1) do return bestS
    for i = 1 to (path.count - 1) do (
        edge = path[i + 1] - path[i]
        length2 = dot edge edge
        if length2 > 1e-24 do (
            t = (dot (vertex - path[i]) edge) / length2
            if t < 0.0 do t = 0.0
            if t > 1.0 do t = 1.0
            proj = path[i] + t * edge
            dist = distance proj vertex
            if dist < bestD do (
                bestD = dist
                bestS = sigma[i] + t * (sigma[i + 1] - sigma[i])
            )
        )
    )
    if bestD > limit do throw ("vertex is not on its radial profile: " + (bestD as string))
    bestS
)

-- Fan from the first vertex, the same triangulation the Python mesh uses.
fn gemFaceTriangles ep = (
    local tris = #()
    local nf = polyop.getNumFaces ep
    local f = 0
    local v = #()
    local i = 0
    for f = 1 to nf do (
        v = polyop.getFaceVerts ep f
        for i = 2 to (v.count - 1) do
            append tris #(v[1], v[i], v[i + 1])
    )
    tris
)

fn gemFitTol ep = (
    local big = 1.0
    local i = 0
    local p = [0, 0, 0]
    local a = 0.0
    local weld = 0.0
    local limit = 0.0
    for i = 1 to (polyop.getNumVerts ep) do (
        p = polyop.getVert ep i
        a = abs p.x
        if (abs p.y) > a do a = abs p.y
        if (abs p.z) > a do a = abs p.z
        if a > big do big = a
    )
    weld = 1e-5 * big
    if weld < 1e-6 do weld = 1e-6
    limit = 1e-4 * big
    if limit < 1e-4 do limit = 1e-4
    #(weld, limit)
)

fn gemMeridian ep tris phi weld = (
    local direction = [gemCosRad phi, gemSinRad phi, 0]
    local normal = [-direction.y, direction.x, 0]
    local pts = #()
    local adj = #()
    local tri = #()
    local corners = #()
    local axis = #()
    local i = 0
    local start = 1
    local startZ = 0.0
    local keys = #()
    local path = #()
    local sigma = #(0.0)
    local rmax = -1.0
    local edge = 1
    local r = 0.0
    for tri in tris do (
        corners = #(
            polyop.getVert ep tri[1],
            polyop.getVert ep tri[2],
            polyop.getVert ep tri[3]
        )
        gemCutTriangle corners normal pts adj direction weld
    )
    if pts.count == 0 do throw "radial cut missed the mesh"
    for i = 1 to pts.count do (
        if (gemHypot pts[i]) <= 1e-8 do append axis i
    )
    if axis.count == 0 do throw "radial cut never reached the axis"
    start = axis[1]
    startZ = pts[start].z
    for i in axis do (
        if pts[i].z > startZ do (
            startZ = pts[i].z
            start = i
        )
    )
    keys = gemProfile start adj pts
    for k in keys do append path pts[k]
    for i = 2 to path.count do
        append sigma (sigma[i - 1] + (distance path[i] path[i - 1]))
    for p in path do (
        r = gemHypot p
        if r > rmax do rmax = r
    )
    for i = 1 to path.count do (
        if (gemHypot path[i]) >= (rmax - weld) do (
            edge = i
            exit
        )
    )
    if sigma[edge] <= 1e-8 do throw "crown edge has no arc length"
    #(path, sigma, sigma[edge], gemHypot path[edge])
)

fn gemFrameHalf ep given = (
    local rmax = 0.0
    local i = 0
    local r = 0.0
    -- `and` would still evaluate the comparison when given is undefined.
    if given != undefined then (
        if given > 0.0 do return given
    )
    for i = 1 to (polyop.getNumVerts ep) do (
        r = gemHypot (polyop.getVert ep i)
        if r > rmax do rmax = r
    )
    if rmax <= 0.0 do throw "the mesh has no vertices"
    rmax
)

fn gemKeepLower ep uv = (
    -- The crown is z >= 0.  Leaving those vertices outside the picture keeps
    -- the drawing on the pavilion when gemPatternFace is "pavilion".
    local out = #()
    local i = 0
    local p = [0, 0, 0]
    for i = 1 to uv.count do (
        p = polyop.getVert ep i
        if p.z < -0.000001 then
            append out uv[i]
        else
            append out [-1, -1, 0]
    )
    out
)

fn gemPlanarUV ep half = (
    local s = gemFrameHalf ep half
    local nv = polyop.getNumVerts ep
    local uv = #()
    local i = 0
    local p = [0, 0, 0]
    for i = 1 to nv do (
        p = polyop.getVert ep i
        append uv [0.5 + p.x / (2.0 * s), 0.5 + p.y / (2.0 * s), 0]
    )
    uv
)

fn gemSurfaceUV ep half = (
    local s = gemFrameHalf ep half
    local tris = gemFaceTriangles ep
    local fit = gemFitTol ep
    local weld = fit[1]
    local limit = fit[2]
    local nv = polyop.getNumVerts ep
    local uv = #()
    local cachePhi = #()
    local cacheMer = #()
    local i = 0
    local c = 0
    local p = [0, 0, 0]
    local rho = 0.0
    local phi = 0.0
    local mer = undefined
    local sigma = 0.0
    local imageR = 0.0
    for i = 1 to nv do (
        p = polyop.getVert ep i
        rho = gemHypot p
        if rho <= 1e-8 then (
            append uv [0.5, 0.5, 0]
        ) else (
            phi = gemRound8 ((atan2 p.y p.x) * pi / 180.0)
            mer = undefined
            for c = 1 to cachePhi.count while mer == undefined do (
                if (abs (cachePhi[c] - phi)) <= 1e-12 do mer = cacheMer[c]
            )
            if mer == undefined do (
                mer = gemMeridian ep tris phi weld
                append cachePhi phi
                append cacheMer mer
            )
            sigma = gemSigmaAt mer[1] mer[2] p limit
            imageR = (sigma / mer[3]) * mer[4]
            append uv [0.5 + (imageR * p.x / rho) / (2.0 * s), 0.5 + (imageR * p.y / rho) / (2.0 * s), 0]
        )
    )
    uv
)

fn gemApplyUV obj uv = (
    local ep = obj.baseObject
    local nv = polyop.getNumVerts ep
    local nf = 0
    local i = 0
    local f = 0
    if (polyop.getNumMaps ep) < 2 do polyop.setNumMaps ep 2
    polyop.setMapSupport ep 1 true
    polyop.setNumMapVerts ep 1 nv
    for i = 1 to nv do polyop.setMapVert ep 1 i uv[i]
    nf = polyop.getNumFaces ep
    for f = 1 to nf do
        polyop.setMapFace ep 1 f (polyop.getFaceVerts ep f)
    update obj
)

fn gemCapUV obj half = (
    -- One map vertex per face corner.  Shared geometric vertices must not
    -- share the picture, or the table texture runs out onto the star facets.
    local ep = obj.baseObject
    local s = gemFrameHalf ep half
    local nf = polyop.getNumFaces ep
    local nv = polyop.getNumVerts ep
    local zmax = -1e9
    local i = 0
    local p = [0, 0, 0]
    local f = 0
    local k = 0
    local verts = undefined
    local corners = 0
    local mv = 1
    local allTop = false
    local allLower = false
    local show = false
    local faceUV = undefined
    for i = 1 to nv do (
        p = polyop.getVert ep i
        if p.z > zmax do zmax = p.z
    )
    if (polyop.getNumMaps ep) < 2 do polyop.setNumMaps ep 2
    polyop.setMapSupport ep 1 true
    for f = 1 to nf do (
        verts = polyop.getFaceVerts ep f
        corners += verts.count
    )
    polyop.setNumMapVerts ep 1 corners
    for f = 1 to nf do (
        verts = polyop.getFaceVerts ep f
        allTop = true
        allLower = true
        for k = 1 to verts.count do (
            p = polyop.getVert ep verts[k]
            if (abs (p.z - zmax)) > 0.0001 do allTop = false
            if p.z >= -0.000001 do allLower = false
        )
        show = false
        if allTop then
            show = true
        else if allLower do
            show = true
        faceUV = #()
        for k = 1 to verts.count do (
            p = polyop.getVert ep verts[k]
            if show then
                polyop.setMapVert ep 1 mv [0.5 + p.x / (2.0 * s), 0.5 + p.y / (2.0 * s), 0]
            else
                polyop.setMapVert ep 1 mv [-1, -1, 0]
            append faceUV mv
            mv += 1
        )
        polyop.setMapFace ep 1 f faceUV
    )
    update obj
)

fn gemPatternMaterial path = (
    local bm = Bitmaptexture filename:path name:"gemopt_pattern_map"
    local mat = undefined
    -- One property missing must not skip the rest.  Tile stays off so a
    -- coordinate past the girdle does not repeat the picture on the pavilion.
    try (bm.coords.realWorldMapSize = false) catch ()
    try (bm.coords.mapChannel = 1) catch ()
    try (bm.coords.u_tiling = 1.0) catch ()
    try (bm.coords.v_tiling = 1.0) catch ()
    try (bm.coords.u_offset = 0.0) catch ()
    try (bm.coords.v_offset = 0.0) catch ()
    try (bm.coords.u_tile = false) catch ()
    try (bm.coords.v_tile = false) catch ()
    mat = Standardmaterial name:"gemopt_pattern" diffuseMap:bm
    showTextureMap mat true
    mat
)

fn gemIsHeadless = (
    local cmd = ""
    if gemoptHeadless == true do return true
    try (cmd = toLower (maxOps.getCommandLine())) catch ()
    if (findString cmd "3dsmaxbatch") != undefined do return true
    if (findString cmd "-silent") != undefined do return true
    false
)

rollout gemPatternRollout "پترن" (
    edittext edtPath "" readOnly:true width:260
    button btnFile "انتخاب فایل" width:260
    checkbutton btnLoose "پهن و رها" width:260 checked:false
    checkbutton btnTight "سفت از بالا" width:260 checked:false

    fn gemEnsurePath = (
        local picked = undefined
        -- `and` does not skip the file test, and doesFileExist on undefined throws
        if gemPatternPath != undefined then (
            if gemPatternPath != "" then (
                if doesFileExist gemPatternPath do return gemPatternPath
            )
        )
        picked = getOpenFileName caption:"پترن" \
            types:"Image (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|All|*.*|"
        if picked == undefined do return undefined
        gemPatternPath = picked
        edtPath.text = picked
        if doesFileExist picked then picked else undefined
    )

    fn gemApplyMode mode = (
        local path = gemEnsurePath()
        local ep = undefined
        local uv = undefined
        if path == undefined or path == "" or gemPatternNode == undefined then (
            gemPatternBusy = true
            btnLoose.checked = false
            btnTight.checked = false
            gemPatternBusy = false
            return undefined
        )
        try (
            ep = gemPatternNode.baseObject
            if gemPatternFace == "pavilion" then (
                gemCapUV gemPatternNode gemHalfWidth
            ) else (
                if mode == "surface" then
                    uv = gemSurfaceUV ep gemHalfWidth
                else
                    uv = gemPlanarUV ep gemHalfWidth
                gemApplyUV gemPatternNode uv
            )
            gemPatternNode.material = gemPatternMaterial path
            redrawViews()
            format "gemopt: pattern % on %\n" mode path
        ) catch (
            format "gemopt: pattern failed (%)\n" (getCurrentException())
            gemPatternBusy = true
            btnLoose.checked = false
            btnTight.checked = false
            gemPatternBusy = false
        )
        OK
    )

    on gemPatternRollout open do (
        if gemPatternPath != undefined do edtPath.text = gemPatternPath
    )

    on btnFile pressed do (
        local picked = getOpenFileName caption:"پترن" \
            types:"Image (*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff)|*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff|All|*.*|"
        if picked != undefined do (
            gemPatternPath = picked
            edtPath.text = picked
            if btnLoose.checked then gemApplyMode "surface"
            else if btnTight.checked do gemApplyMode "planar"
        )
    )

    on btnLoose changed state do (
        if gemPatternBusy do return false
        if state == false do return false
        gemPatternBusy = true
        btnTight.checked = false
        gemPatternBusy = false
        gemApplyMode "surface"
    )

    on btnTight changed state do (
        if gemPatternBusy do return false
        if state == false do return false
        gemPatternBusy = true
        btnLoose.checked = false
        gemPatternBusy = false
        gemApplyMode "planar"
    )
)

fn gemShowPattern obj = (
    gemPatternNode = obj
    gemPatternBusy = false
    -- This run's picture replaces whatever the previous stone left behind.
    if gemPattern != undefined then (
        gemPatternPath = gemPattern
    ) else (
        gemPatternPath = ""
    )
    if gemIsHeadless() then (
        format "gemopt: pattern rollout skipped\n"
    ) else (
        try (destroyDialog gemPatternRollout) catch ()
        try (
            createDialog gemPatternRollout 280 160
        ) catch (
            format "gemopt: pattern rollout failed (%)\n" (getCurrentException())
        )
    )
)

-- ---------------------------------------------------------------- main ----

(
    local path = gemFile
    if path == undefined do
        path = getOpenFileName caption:"gemopt stone" types:"gemopt stone (*.gem)|*.gem|All|*.*|"
    -- "then", not "do": the `do` form of if takes no else branch, and pairing
    -- the two is a syntax error that stops the whole script before a single
    -- vertex is built.
    if path == undefined then (
        format "gemopt: cancelled\n"
    ) else (
        local data = gemReadFile path
        local verts = data[1]
        local faces = data[2]

        if verts.count == 0 or faces.count == 0 then (
            format "gemopt: % has no geometry\n" path
        ) else (
            local names = data[3]
            local ior = gemDefault data[4] 1.52
            local objName = gemDefault gemName "gemopt_stone"
            local obj = gemBuild verts faces objName
            local rows = gemRowsToMatIDs obj names
            local rowCount = rows.count
            gemAssignMaterial obj (gemDefault gemMaterial "vray") ior rows
            gemAddLight gemIES

            -- A .max file can only be written from inside 3ds Max, so if one
            -- was asked for this is where it happens.
            local savePath = gemSaveMax
            if savePath == "auto" do
                savePath = (getFilenamePath path) + (getFilenameFile path) + ".max"
            if savePath != undefined do (
                try (
                    saveMaxFile savePath quiet:true
                    format "gemopt: saved %\n" savePath
                ) catch (
                    format "gemopt: could not save the .max file (%)\n" \
                        (getCurrentException())
                )
            )

            select obj
            try (max zoomext sel all) catch ()
            format "gemopt: built '%' with % vertices, % facets, % rows\n" \
                objName verts.count faces.count rowCount
            gemShowPattern obj
        )
    )
)
