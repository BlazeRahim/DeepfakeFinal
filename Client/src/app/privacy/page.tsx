'use client';
import React, { useEffect, useRef, useState } from 'react'

const Page = () => {
  const numbers = ['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve']
  const headings: string[] = [
    "Information We Collect About You",
    "How We Collect Your Data",
    "How We Use Your Information",
    "Data Retention",
    "Your Rights Under GDPR",
    "Data Security",
    "Third-party Links",
    "Data Transfers",
    "Data Processing Details",
    "Exercising Your GDPR Rights",
    "Changes to This Privacy Policy",
    "Contact Us"
  ];

  const [rightpx, setrightpx] = useState(0)


  const scroll = (className: string): void => {
    const target = document.querySelector(`.${className}`) as HTMLElement | null;
    if (target) {
      console.log(target);
      target.scrollIntoView({ behavior: 'smooth' });
    }
  };
  





  useEffect(() => {
    calculateValues();
    // Add event listener to window resize event
    window.addEventListener('resize', () => {// Update the width state
      calculateValues(); // Call the function to perform calculations
    });

    // Clean up the event listener when the component unmounts
    return () => {
      window.removeEventListener('resize', () => {
        calculateValues();
      });
    };
  }, []);


  const calculateValues = () => {
    if (window.innerWidth > 1200) {
      // Get the total width of the viewport in pixels
      const totalWidth = window.innerWidth;

      // Subtract 1200px from the total width
      const result = totalWidth - 1200;

      // Calculate half of the result
      const halfValue = result / 2;
      setrightpx(halfValue)

      console.log('Total Width:', totalWidth, 'px');
      console.log('Result after subtracting 1200px:', result, 'px');
      console.log('Half of the Result:', halfValue, 'px');
    } else {
      console.log('Screen width is not less than 768px.');
    }
  }

  return (
    <div className=' in-h-[100vh]  overflow-x-hidden flex items-center justify-start flex-col md:w-[100%] base:w-[100%]'>
      <div className='text-pop  w-[min(90vw,1200px)] h-auto flex relative '>

        <div className='base:w-[100%] lg:w-[calc(100%_-_380px)]  min-h-[100vh] flex  items-center flex-col '>

          <div className='w-full h-[170px] flex justify-center items-center flex-col gap-2'>
            <h1 className='w-[90%] text-left text-4xl font-[600] tracking-wide'>Privacy Policy</h1>
            <p className='w-[90%] text-left  text-muted-foreground text-[0.8rem] tracking-wider'>Effective Date: 6.01.2024
            </p>
          </div>

          <div className='w-[90%] h-auto flex flex-col gap-5'>
            <h2 className='w-[87%] font-[600] text-lg tracking-light'>This Privacy policy will help you understand how we collect , use , and share your personal information</h2>

            <div className='w-full h-auto flex flex-col gap-1' >
              <h2 className='w-[57%] font-[600] text-lg tracking-light'>Privacy Policy</h2>
              <p className='text-[0.94rem] w-[95%]'>Veriface (&quot;<span className='font-[620]'>we</span>&quot;, &quot;<span className='font-[620]'>us</span>&quot;, or &quot;<span className='font-[620]'>our</span>&quot;) operates the <span className='text-[#5066E3] hover:underline cursor-pointer'>https://veriface-app.vercel.app</span> (&quot;<span className='font-[620]'>Service</span>&quot;) and is committed to protecting the privacy and security of your personal information. This Privacy Policy outlines how we handle your personal information in relation to our deepfake video detection service. We adhere to the General Data Protection Regulation (<span className='font-[620]'>GDPR</span>) and outline our practices concerning the collection, use, and sharing of your data, as well as your rights in relation to this data.</p>

            </div>




            <div className='w-full flex flex-col mt-[40px] gap-[40px] mb-[100px]'>



              <div className='w-full flex flex-col gap-2 one'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>1. Information We Collect About You  </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We collect various types of information from our users, including:</p>
                  <ul className='list-disc ml-[20px] flex flex-col gap-1'>
                    <li>We exclusively collect the videos uploaded by the user for deepfake video detection purposes.</li>
                    <li>Our data collection includes gathering personal information such as name, email address, and contact number from the user.</li>
                    <li>Additionally, technical data like device information and IP address is collected to ensure efficient operation of our deepfake detection service..</li>
                  </ul>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 two'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>2. How We Collect Your Data
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We obtain your personal data through:</p>
                  <ul className='ml-[20px] list-disc flex flex-col gap-1'>
                    <li>we prioritize ensuring the accuracy and integrity of our deepfake video detection service.</li>
                    <li>The data collected from these sources is solely for the purpose of detecting and analyzing potential deepfake content.</li>
                    <li>Rest assured, the data obtained from these sources is utilized exclusively for deepfake detection and analysis, and is not used for any other activities.</li>
                  </ul>
                </div>
              </div>

              <div className='w-full flex flex-col gap-2 three'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>3. How We Use Your Information
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We use your information for purposes such as:</p>
                  <ul className='ml-[20px] list-disc flex flex-col gap-1'>
                    <li>We utilize the information provided by users to verify whether the submitted video content is authentic or edited using deepfake technology.</li>
                    <li>Complying with legal obligations.</li>
                    <li>Improving our Service and products.</li>
                  </ul>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 four'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>4. Data Retention
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We will only retain your personal data for as long as necessary to fulfill the purposes we collected it for, including for the purposes of satisfying any legal, accounting, or reporting requirements.  </p>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 five'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>5. Your Rights Under GDPR
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>Under GDPR, you have the right to:  </p>
                  <ul className='ml-[20px] list-disc flex flex-col gap-1'>
                    <li>Under GDPR, you have various rights including access to your personal data, correction of inaccuracies, erasure of personal data, and objection to processing.</li>
                  </ul>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 six'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>6. Data Security
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We have implemented appropriate security measures to prevent your personal data from being accidentally lost, used, or accessed in an unauthorized way, altered, or disclosed. </p>
                </div>
              </div>

              <div className='w-full flex flex-col gap-2 seven'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>7. Third-party Links
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>This Service may include links to third-party websites. Clicking on those links may allow third parties to collect or share data about you. We do not control these third-party websites and are not responsible for their privacy statements.  </p>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 eight'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>8. Data Transfers
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We do not transfer your personal data outside the European Economic Area (EEA) unless there are adequate controls in place, including the security of your data and other personal information.  </p>
                </div>
              </div>


              <div className='w-full flex flex-col gap-2 nine'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>9. Data Processing Details
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We do not store documents while providing translation services unless you make changes to the translated document and consent to its storage. Similarly, for speech-to-text services, only the text converted from speech is stored with your consent.  </p>
                </div>
              </div>

              <div className='w-full flex flex-col gap-2 ten'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>10. Exercising Your GDPR Rights
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>To exercise any of your rights under GDPR, please contact us at [Your Contact Information]. We may need to request specific information from you to help us confirm your identity and ensure your right to access your personal data or to exercise any of your other rights.
                  </p>
                </div>
              </div>

              <div className='w-full flex flex-col gap-2 eleven'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>11. Changes to This Privacy Policy  

                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>We may update our Privacy Policy from time to time. We will notify you of any changes by posting the new Privacy Policy on this page and updating the &quot;Effective Date&quot; at the top of this Privacy Policy.  
                  </p>
                </div>
              </div>

              <div className='w-full flex flex-col gap-2 twelve'>
                <h2 className='base:[100%] md:w-[57%] font-[600] text-[1.3rem] tracking-light'>12. Contact Us  
                </h2>
                <div className='flex flex-col text-[0.94rem] gap-[5px]'>
                  <p>For more information about our privacy practices, if you have questions, or if you would like to make a complaint, please contact us by email at [hi@thevotum.com] or by mail at [404-H DJAC, IIT Kanpur, Kanpur, India].  
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div style={{ right: `${rightpx}px` }} className='base:hidden lg:flex fixed top-[8rem] w-[330px] h-[400px] flex-col justify-start items-center gap-2'>
          <h1 className='w-full text-xl font-[620] tracking-normal '>Table of content</h1>
          <div className='w-[88%]'>
            <ol className='text-[0.82rem] font-[500] text-muted-foreground flex flex-col gap-[0.32rem] cursor-pointer ulcontent'>
              {headings.map((heading, index) => (
                <li onClick={(e)=>scroll(numbers[index])} className='hover:text-foreground dark:hover:text-foreground hover:font-[600]' key={index}>{heading}</li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Page