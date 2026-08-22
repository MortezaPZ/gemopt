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
*/

global gemFile
global gemName
global gemMaterial
global gemIES
global gemSaveMax

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
        )
    )
)
