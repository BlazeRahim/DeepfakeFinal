'use client';
import { createFFmpeg, fetchFile } from "@ffmpeg/ffmpeg/src/index.js"
const FF = createFFmpeg({
    log: false,
    corePath: "https://unpkg.com/@ffmpeg/core@0.10.0/dist/ffmpeg-core.js",
  });

export const getFirstFrameImageURL = async (file) => {
    if (!FF.isLoaded()) await FF.load();
    const startTimeInSecs = "00:00:00.1";
    FF.FS("writeFile", file.name, await fetchFile(file));

    try {
      console.log(`Generating the first frame`);
      await FF.run(
        "-ss",
        startTimeInSecs,
        "-i",
        file.name,
        "-t",
        "00:00:1.000",
        "-vf",
        "scale=1280:-1",
        "firstFrame.png"
      );
      const data = FF.FS("readFile", "firstFrame.png");
      let blob = new Blob([data.buffer], { type: "image/png" });
      // Convert the Blob to a data URL
      const dataURI = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });
      FF.FS("unlink", "firstFrame.png");
      return dataURI
    } catch (error) {
      console.error(`Error generating the first frame:`, error);
      return null;
    }
  };