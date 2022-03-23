import React from 'react'
import { useState } from 'react';
import { AiFillHome } from 'react-icons/ai'
import { BsFillPeopleFill } from 'react-icons/bs'
import { BiShow } from 'react-icons/bi'
import './nav.css';

const Nav = () => {
    const [activeNav, setactiveNav] = useState('#')
    return (
        <nav>
            <a class="nav-link" href="#"><AiFillHome class="icons"/></a>
            <a class="nav-link" href='#card'><BiShow class="icons"/></a>
            <a class="nav-link" href="#teams"><BsFillPeopleFill class="icons"/></a>
        </nav>
        )
    }

export default Nav