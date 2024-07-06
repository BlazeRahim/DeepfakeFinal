'use client'
import React, { useEffect } from "react";
import * as helpers from "./utils/helpers";
import LinearIndeterminate from "./loadingAnimation"


export default function RangeInput({
  thumbNails,
  rEnd,
  rStart,
  handleUpdaterStart,
  handleUpdaterEnd,
  loading,
  control,
  videoMeta,
}) {
  let RANGE_MAX = 100;
  useEffect(() => {
    console.log(videoMeta)
  }, [videoMeta])
  if (thumbNails.length === 0 && !loading) {
    return null;
  }
  if (loading) {
    return (
      <center style={{ justifyContent: "center" }}>
        <LinearIndeterminate />
      </center>
    );
  }
  return (
    <>
      <div className="w-[100%] base:hidden bl:flex">
        <div className="image_box flex w-[100%] justify-between  max-w-[100%]">
          {thumbNails.map((imgURL, id) => (
            <img src={imgURL} className="flex-1 select-none object-contain min-h-[60px]" alt={`sample_video_thumbnail_${id}`} key={id} />
          ))}
          <div
            className="clip_box"
            style={{
              width: `calc(${rEnd - rStart}% )`,
              left: `${rStart}%`,
            }}
            data-start={helpers.toTimeString(
              (rStart / RANGE_MAX) * videoMeta.duration,
              false
            )}
            data-end={helpers.toTimeString(
              (rEnd / RANGE_MAX) * videoMeta.duration,
              false
            )}
          >

            <span className="clip_box_des"></span>
            <span className="clip_box_des"></span>
          </div>

          <input
            className="range"
            type="range"
            min={0}
            max={RANGE_MAX}
            onInput={handleUpdaterStart}
            value={rStart}
          />
          <input
            className="range"
            type="range"
            min={0}
            max={RANGE_MAX}
            onInput={handleUpdaterEnd}
            value={rEnd}
          />
        </div>
      </div>
      <div className="w-[100%] base:flex bl:hidden">
        <div className="image_box flex w-[100%] justify-between overflow-hidden max-w-[100%]">
          {thumbNails.map((imgURL, id) => (
            <img src={imgURL} className="flex-1 select-none object-contain min-h-[60px]" alt={`sample_video_thumbnail_${id}`} key={id} />
          ))}
          <div
            className="clip_box"
            style={{
              width: `calc(${rEnd - rStart}% )`,
              left: `${rStart}%`,
            }}
            data-start={helpers.toTimeString(
              (rStart / RANGE_MAX) * videoMeta.duration,
              false
            )}
            data-end={helpers.toTimeString(
              (rEnd / RANGE_MAX) * videoMeta.duration,
              false
            )}
          >

            <span className="clip_box_des"></span>
            <span className="clip_box_des"></span>
          </div>

          <input
            className="range"
            type="range"
            min={0}
            max={RANGE_MAX}
            onInput={handleUpdaterStart}
            value={rStart}
          />
          <input
            className="range"
            type="range"
            min={0}
            max={RANGE_MAX}
            onInput={handleUpdaterEnd}
            value={rEnd}
          />
        </div>
      </div>
      {control}
    </>
  );
}
