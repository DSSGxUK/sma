# SMA Project - DSSGx Summer 2021

## Project Overview

SMA (The Superintendency of the Environment) is the environmental regulator of Chile. SMA receives environmental complaints from citizens and other organizations, and must determine if the complaint should get an inspection or not. If the inspection reveals any infractions of environmental regulations, a sanction procedure is initiated and can lead to a monetary fine.
SMA current process of review

## Problem Statement
SMA Complaint Process
The SMA Organization had a straight forward process for handling complaints filed by citizens. The problem with this system is the complaints are filed using a paper form, and SMA has to transcribe this information before entering it into their database systems. This transcription process can be very time consuming and is error prone since a transcriber could introduce human error. For this reason, SMA recently launched an online platform which allows citizens to submit complaints directly to the organization. This online platform also makes the complaints procedure accessible to more people in more remote locations, enabling SMA to get more reliable information to help them protect ecosystems and public health.

## The Problem
Since the introduction of the new online platform, the volume of incoming complaints has quadrupled, and hence carrying out inspections and sanctions for the complaints is getting increasingly difficult. This project aims to build a machine learning system that serves two main purposes:

It determines if the complaint is relevant or not to SMA, in which case an analyst can launch an inspection or redirect the complaint as required. Once an inspection is carried out, the complaint is either archived if there is not enough evidence to substantiate the complaint, otherwise, a sanction procedure is initiated.

It attempts to determine what the sanction of a complaint should be, based on historical complaints, inspections and sanctions.

## Solution Approach
This project proposes two solutions to solve the above problem:

 1. A model to predict the 3 class relevance of each complaint as either being relevant, low-quality, or suitable to be forwarded to another authority
 2. A model to predict the severity of each complaint. 


Please see our video for more information: https://www.youtube.com/watch?v=QO1cLMXL8Fo
