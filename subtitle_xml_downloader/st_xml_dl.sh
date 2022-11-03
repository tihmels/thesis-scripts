#!/bin/zsh
#
# Download tagesschau subtitles.
#



from='2022-06-09'
to='2022-09-17'

startdate=$(date -j -f '%Y-%m-%d' "${from}" +%Y%m%d)
enddate=$(date -j -f '%Y-%m-%d' "${to}" +%Y%m%d)

out="subtitles"
base="https://www.tagesschau.de"

if [ ! -d "${out}" ]; then
	mkdir "${out}"
fi

i=0
currdate=${startdate}

while [ "${currdate}" != "${enddate}" ]; do
    currdate=$(date -j -f '%Y%m%d' -v+1d ${currdate} +%Y%m%d) # get $i days forward
    
    echo "${currdate}"

    url=$(curl -s "${base}/multimedia/video/videoarchiv2~_date-${currdate}.html" | pup ".dachzeile:contains(\"20:00 Uhr\") + .headline a:contains(\"tagesschau\") attr{href}" | grep "/ts-" | head -n 1)
	id=$(echo ${url} | grep -o '[0-9]\+')

	json=$(curl -s "${base}${url}" \
			| pup --plain "div [data-ts_component="ts-mediaplayer"] attr{data-config}")
	
	suburl=$(echo $json | jq -r '.[] | ._subtitleUrl' | head -n 1)

	subtitle=$(curl -s "${base}${suburl}" > ${out}/TV-${currdate}-${id}.xml )

    i=$(( i + 1 ))
done

exit 1