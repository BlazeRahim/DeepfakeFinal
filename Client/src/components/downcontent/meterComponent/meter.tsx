'use client'
import React from 'react'
import MeterBar from './ReviewsBar'
const Meter = () => {
  return (
    <div className='base:w-full tab:w-[50%] h-full flex justify-center  items-center'>
      <div className='base:w-[75%] tab:w-[100%] tab:h-[95%] flex justify-center  items-center'>
        <MeterBar className="borv" score={93} />
      </div>
    </div>
  )
}

export default Meter