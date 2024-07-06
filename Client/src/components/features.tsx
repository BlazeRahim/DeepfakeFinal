'use client';
import React from 'react'
import Image from 'next/image'
import { motion } from "framer-motion";
const Features = () => {
  return (
    <div className='feature min-w-[min(1300px,90%)] w-[min(1300px , 90vw)] relative  base:max-w-[90vw] bl:max-w-[1700px] flex flex-col items-center py-[20px]  base:mt-[30px] bl:mt-[60px] overflow-hidden'>
      <div className='flex bl:w-[80%] bl:px-[36px]   flex-col base:gap-[20px] bl:gap-[30px] justify-center items-center'>
        <h2 className='text-primary font-[600] text-md'>VERIFACE FEATURES</h2>
        <h1 className='base:text-2xl bl:text-[2.5rem] overflow-visible z-[0] relative text-center font-[650] tracking-wide'>
          Veriface's goal is accurate deepfake detection using AI, achieving
          <span className='snatch base:py-[0px] bl:py-[25px]  relative base:mx-[10px] bl:mx-[18px] z-[1000000] text-white text-center'>
            93%
          </span>
          accuracy with facial expression analysis features.</h1>
      </div>
      <div className='w-full my-[80px] mt-[100px] flex justify-center items-start gap-[35px]'>
        <div className='w-[250px] mt-[-30px] gap-[55px] flex flex-col items-end'>
          <motion.div initial={{ opacity: 0 , scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.1 }} className='w-full flex flex-col gap-[7px] items-end justify-start '>
            <h1 className='text-2xl text-right font-[600]'>User-friendly Interface</h1>
            <p className='text-right text-sm text-muted-foreground'>Offers an intuitive interface, making it easy for users to navigate and utilize its powerful detection capabilities.</p>
          </motion.div>
          <motion.div initial={{ opacity: 0, scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.2 }} className='w-full flex flex-col gap-[7px] items-end justify-start'>
            <h1 className=' text-right text-2xl font-[600]'>Link-based Video Detection</h1>
            <p className='text-right text-sm text-muted-foreground'>Detect deepfakes in videos from YouTube, Instagram, and Drive by simply providing the video link.</p>
          </motion.div>
          <motion.div initial={{ opacity: 0, scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.3 }} className='w-full flex flex-col gap-[7px] items-end justify-start' >
            <h1 className='text-2xl text-right font-[600]'>Real-time Monitoring</h1>
            <p className=' text-right text-sm text-muted-foreground'>Users can actively monitor for deepfake content in real-time, ensuring timely detection and response.</p>
          </motion.div>
        </div>
        <Image src='/images/Groupfinal.png' alt="meow" width={600} height={600} className='base:hidden bl:flex' />
        {/* <img src='/images/1024V.png' alt='qwerty' className='base:flex hidden  w-[400px] h-[500px]'  /> */}
        <div className='w-[250px] mt-[-30px] gap-[55px] flex flex-col items-end'>
          <motion.div initial={{ opacity: 0, scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.1 }}  className='w-full flex flex-col gap-[7px] items-start justify-start '>
            <h1 className='text-left text-2xl font-[600]'>Verify Social media videos</h1>
            <p className='text-left text-sm text-muted-foreground'>Detect deepfakes in videos from YouTube, Instagram, and Drive by simply providing the video link.</p>
          </motion.div>
          <motion.div initial={{ opacity: 0, scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.2 }}  className='w-full flex flex-col gap-[7px] items-start justify-start'>
            <h1 className='text-2xl text-left font-[600]'>Confidence Score</h1>
            <p className='text-left text-sm text-muted-foreground'> The app provides a confidence score for each detection, allowing users to gauge the likelihood of content being a deepfake.</p>
          </motion.div>
          <motion.div initial={{ opacity: 0, scale:0 }}
            viewport={{ once: true }}
            whileInView={{ opacity: 1, scale:1 }}
            transition={{ duration: 0.7, type: "Spring", bounce: 0.4 , delay:0.3 }}  className='w-full flex flex-col gap-[7px] items-start justify-start' >
            <h1 className='text-left text-2xl font-[600]'>Facial Expression Analysis</h1>
            <p className=' text-left text-sm text-muted-foreground'>analyzes facial expressions to detect inconsistencies indicative of deepfake manipulation.</p>
          </motion.div>
        </div>
      </div>
    </div>
  )
}

export default Features