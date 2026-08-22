/*  gemopt -- export the selected stone out of 3ds Max
    ------------------------------------------------------------------
    Writes the selected Editable Poly as a .gem file (facets kept intact) and,
    alongside it, the same vertex table the existing tool sends to Excel.

    That gives two things:
      * an existing hand-cut model can be measured and scored by the Python
        side without being rebuilt, and
      * the familiar Excel round-trip keeps working.

    Run from the Scripting menu with the stone selected, or head-less:
        3dsmaxbatch.exe export_stone.ms -mxsString outFile:"C:\path\mine.gem"

    Optional globals:
        outFile   where to write        (a save dialog opens if unset)
        outNode   name of the node to export instead of the selection
*/

global outFile
global outNode

fn gemFacetGuess normal centreZ z = (
    -- A useful label per facet, worked out from where it sits and which way
    -- it faces.  It is only a hint for reading the file later; the Python
    -- side never relies on these names.
    local nz = normal.z
    if (abs nz) < 0.2 then "girdle"
    else if z > centreZ then (
        if (abs nz) > 0.95 then "table" else "crown"
    )
    else (
        if (abs nz) > 0.95 then "culet" else "pavilion"
    )
)

fn gemExport node path = (
    local ep = node.baseObject
    local nv = polyop.getNumVerts ep
    local nf = polyop.getNumFaces ep
    if nv == 0 or nf == 0 do throw "the selected object has no geometry"

    local zs = for i = 1 to nv collect (polyop.getVert ep i).z
    local centreZ = 0.0
    for z in zs do centreZ += z
    centreZ /= nv

    local fs = createFile path
    if fs == undefined do throw ("cannot write " + path)

    format "# gemopt stone v1 (exported from 3ds Max)\n" to:fs
    format "scale 1.00000000\n" to:fs
    format "facets %\n" nf to:fs
    format "verts %\n" nv to:fs

    for i = 1 to nv do (
        local v = polyop.getVert ep i
        format "v % % %\n" v.x v.y v.z to:fs
    )

    for f = 1 to nf do (
        local idx = polyop.getFaceVerts ep f
        local n = polyop.getFaceNormal ep f
        local zc = 0.0
        for i in idx do zc += (polyop.getVert ep i).z
        zc /= idx.count

        format "f %" (gemFacetGuess n centreZ zc) to:fs
        for i in idx do format " %" i to:fs
        format "\n" to:fs
    )
    close fs

    -- the Excel-facing vertex table, same idea as the existing tool
    local csvPath = (getFilenamePath path) + (getFilenameFile path) + "_vertices.csv"
    local cs = createFile csvPath
    if cs != undefined do (
        format "vertex,x,y,z\n" to:cs
        for i = 1 to nv do (
            local v = polyop.getVert ep i
            format "%,%,%,%\n" i v.x v.y v.z to:cs
        )
        close cs
        format "gemopt: vertex table -> %\n" csvPath
    )

    #(nv, nf)
)

-- ---------------------------------------------------------------- main ----

(
    local node = undefined
    if outNode != undefined then node = getNodeByName outNode
    else if selection.count == 1 then node = selection[1]

    if node == undefined then (
        format "gemopt: select exactly one stone first (or set outNode)\n"
    )
    else (
        local work = node
        if (classOf work.baseObject) != Editable_Poly do (
            work = copy node
            work.name = node.name + "_gemopt_tmp"
            convertTo work Editable_Poly
            format "gemopt: converted a copy to Editable Poly for export\n"
        )

        local path = outFile
        if path == undefined do
            path = getSaveFileName caption:"Export stone" types:"gemopt stone (*.gem)|*.gem|"

        if path == undefined then format "gemopt: cancelled\n"
        else (
            try (
                local info = gemExport work path
                format "gemopt: wrote % (% verts, % facets)\n" path info[1] info[2]
            ) catch (
                format "gemopt: export failed -- %\n" (getCurrentException())
            )
        )

        if work != node do delete work
    )
)
