#!/bin/sh
#
#     ^ ^
#   <(O_o)>
#    ( . )
# ----"-"------->>
# j.fuentesaguilar
#
# DESI DR1 spectra of CatWISE-like sources from Astro Data Lab, written to data/desi_catwise/desi_catwise_like.csv.
# Run from the top of the repository. The query runs as an asynchronous job and can take a while.

Q="SELECT x.ra1 AS ra, x.dec1 AS dec, z.z, z.zwarn, z.spectype, z.program, z.desi_target, z.bgs_target, z.mws_target, a.w1mpro, a.w2mpro FROM desi_dr1.x1p5__zpix__allwise__source AS x JOIN desi_dr1.zpix AS z ON z.id = x.id1 JOIN allwise.source AS a ON a.source_id = x.id2 WHERE z.zcat_primary = 'true' AND a.w1mpro < 16.6 AND a.w1mpro - a.w2mpro > 0.6 AND (z.glat > 30 OR z.glat < -30)"

mkdir -p data/desi_catwise
JOB=$(curl -s -i "https://datalab.noirlab.edu/tap/async" --data-urlencode "REQUEST=doQuery" --data-urlencode "LANG=ADQL" \
      --data-urlencode "FORMAT=csv" --data-urlencode "QUERY=$Q" | grep -i "^location" | awk '{print $2}' | tr -d '\r')
echo "job $JOB"
curl -s -X POST "$JOB/phase" -d "PHASE=RUN" > /dev/null

until p=$(curl -s "$JOB/phase"); [ "$p" = "COMPLETED" ] || [ "$p" = "ERROR" ] || [ "$p" = "ABORTED" ]; do
    sleep 30
done
echo "$p"
if [ "$p" = "COMPLETED" ]; then
    curl -sL "$JOB/results/result" -o data/desi_catwise/desi_catwise_like.csv
else
    curl -s "$JOB/error"
fi
