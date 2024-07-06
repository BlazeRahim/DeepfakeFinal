const scroll = (string) => {
    let target = document.querySelector(`.${string}`);
    console.log(target);
    target.scrollIntoView({behavior:"smooth"});
}

export default scroll;