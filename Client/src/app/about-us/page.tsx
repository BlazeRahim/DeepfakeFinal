import React from 'react';
import Navbar from '@/components/Navbar'
function AboutUs() {
    return (
        <div className='flex justify-center overflow-hidden flex-col w-full items-center'>
            <div className='base:w-full bl:w-[min(1400px,90%)] flex flex-col items-center '>
                <Navbar />
                <div className="min-w-[80%]  w-[80%] flex flex-col base:py-[65px] tab:py-[80px]">
                    <h2 className="text-3xl font-semibold mb-8">Welcome to Veriface</h2>
                    <p className="mb-4">Your trusted tool in the fight against deepfake videos. In an era where digital manipulation has become increasingly sophisticated, Veriface stands as a beacon of authenticity and truth.</p>
                    <p className="mb-4">At Veriface, we leverage cutting-edge technology, including Generative Adversarial Networks (GANs) and Convolutional Neural Networks (CNNs), to detect and identify deepfake videos with unparalleled accuracy. Our state-of-the-art algorithms analyze intricate details within video content, distinguishing between genuine footage and deceptive fabrications.</p>
                    <p className="mb-4">We understand the critical importance of combating misinformation and safeguarding the integrity of visual media. Whether you're a journalist, a content creator, or an individual concerned about the spread of misinformation, Veriface empowers you to verify the authenticity of video content with confidence.</p>
                    <p className="mb-4">Our mission is clear: to provide users with a reliable and accessible solution for detecting deepfake videos, thereby preserving trust and accountability in the digital realm. With Veriface, you can rest assured that truth and authenticity are always within reach.</p>
                    <p className="mb-4">Join us in our quest to combat deepfake technology and uphold the standards of integrity and truthfulness in visual media. Together, we can make a difference in the fight against misinformation.</p>
                </div>
            </div>
        </div>
    );
}

export default AboutUs;