# Project Write-up: CaseTrack - Judiciary Management System

## What CaseTrack is

CaseTrack is a Judiciary Management System that I have been building and debugging as my university DBMS project. The idea is simple: courts deal with a lot of connected information, and keeping it on paper or in scattered files makes it hard to find anything. CaseTrack keeps that information in one relational database and gives people a web interface to work with it.

The main things CaseTrack manages are users, courts, judges, lawyers and cases. These are naturally related to each other. A case is heard in a court, a judge is assigned to it, lawyers represent the parties, and users of the system need to log in to see or change the records. That web of relationships is exactly why a relational database is a good fit for this kind of system.

It is still a project I'm developing and debugging, so I won't call it fully finished.

## The tech stack

CaseTrack has three layers:

1. **Frontend:** HTML for the structure of the pages, CSS for the styling, and JavaScript for the interactive parts, mainly navigation between the different sections such as Cases.
2. **Backend:** Python with the Flask framework. Flask receives requests from the browser through routes, runs the logic, talks to the database and sends results back.
3. **Database:** Microsoft SQL Server, which I manage through SQL Server Management Studio (SSMS). The tables for the different entities live here, along with the relationships between them.

## My role

My part was the frontend and the database integration. A website for the project already existed, and my job was to make it actually work with real data: connecting the existing pages to SQL Server through Flask so that what you click in the browser ends up as a real database operation. I tried to keep the existing structure and technologies of the project instead of throwing everything away and rewriting it.

## The hardest problems

The biggest headache was navigation on the frontend. At one point, clicking a section like "Cases" did not show the page or view it was supposed to show. On the surface it looks like a tiny bug, but fixing it meant understanding how the JavaScript event handlers decide which view is visible, and how those views relate to the Flask routes behind them.

The second challenge was the connection between layers. A button in the browser has to call the right backend route, the route has to run the right query, and the query has to match the actual table structure in SQL Server. If any one of those three is wrong, the feature fails, and the error might show up in a completely different layer than where the real mistake is. So I had to learn to debug across layers: check the browser, check the Flask route, then check the query in SSMS.

## What I learned

- How the frontend, backend and database layers fit together in a real application.
- Why routes and JavaScript event handlers matter so much; most of my "the page doesn't work" bugs came from these.
- How a relational database supports a real-world management system, with entities like courts, judges, lawyers and cases linked to each other.
- That debugging is a normal part of building software, not a sign that you're bad at it.

## Why it's the project I'm most proud of

CaseTrack is my favourite project because it brings together almost everything I'm learning at the same time: a web interface, a Python Flask backend and a SQL Server database. Before this, I studied each technology on its own. This project forced me to see them as one system and to troubleshoot problems that cross the boundaries between them.

---

# Other practical work

## SQL practice and database design (coursework)

Alongside CaseTrack, I've done a lot of SQL practice in SQL Server for my DBMS course. That includes creating tables, inserting records, working with logins and database users, granting and revoking permissions, and generally practising DDL, DML and DCL queries. This is coursework and practice, not a separate product, but it is where most of my SQL confidence comes from. The main things I learned are how relational databases are structured, how to manipulate data properly, and how database access control works.

## EMU8086 assembly exercises (coursework)

For my Assembly Language course I wrote small programs in the EMU8086 emulator. They use registers, arrays, loops, flags, different addressing modes and input/output macros. My role was simple: write the programs, test them and fix them when they didn't work, which happened a lot, especially with syntax and macro errors. These are learning exercises, not a portfolio project, but they taught me how a program really executes at a low level, how registers and memory addressing work, and how to debug when the error messages aren't very friendly.

## Final year project

I haven't finalised my final year project idea yet.
