

const OutputVideo = ({ videoSrc }) => {
  return videoSrc ? (
    <>
      <article className="w-full  flex flex-col items-center justify-center">
        <div className="w-full h-[300px] ">
          <video crossOrigin="anonymus" src={videoSrc} autoPlay controls muted className="w-full h-full"></video>
        </div>
      </article>
       
    </>
  ) : null;
};
  
  export default OutputVideo;



  