# My Study Notes (DBMS, SQL Server, EMU8086 Assembly, Parallel and Distributed Computing)

These are my own short revision notes from my courses. I write notes like this before quizzes: key points first, then a small example.

## 1. DBMS and SQL in SQL Server

**Relational database basics.** Data is stored in tables (relations). Each row is a record, each column is an attribute. A primary key uniquely identifies each row, and a foreign key links a row to a row in another table. In CaseTrack, for example, a case would point to the court it belongs to through a foreign key.

**The three kinds of SQL commands I practise the most:**

- **DDL (Data Definition Language):** defines the structure. `CREATE TABLE`, `ALTER TABLE`, `DROP TABLE`.
- **DML (Data Manipulation Language):** works with the data inside tables. `INSERT`, `UPDATE`, `DELETE`, `SELECT`.
- **DCL (Data Control Language):** controls who can do what. `GRANT`, `REVOKE`, `DENY`.

Small example in T-SQL:

```sql
CREATE TABLE Court (
    CourtID INT PRIMARY KEY,
    CourtName VARCHAR(100) NOT NULL,
    City VARCHAR(50)
);

INSERT INTO Court (CourtID, CourtName, City) VALUES (1, 'District Court', 'Lahore');

SELECT CourtName FROM Court WHERE City = 'Lahore';
```

**Logins vs users in SQL Server.** This confused me at first. A *login* is at the server level: it lets you connect to the SQL Server instance. A *user* is at the database level: it is mapped to a login and decides what you can do inside one specific database. So the usual order is: create the login, create a user for it in the database, then grant permissions to that user.

```sql
CREATE LOGIN student_login WITH PASSWORD = '<choose-a-strong-one>';
CREATE USER student_user FOR LOGIN student_login;
GRANT SELECT ON Court TO student_user;
```

**Why access control matters.** In a system like a judiciary database, not everyone should be allowed to change case records. Permissions make sure people can only do what their role needs.

**SSMS tip.** I test every query in SSMS on its own before putting it into my Flask code. If the query works in SSMS but fails in the app, the bug is in the app layer, not the SQL.

## 2. Assembly Language with EMU8086

**Registers.** The 8086 has general purpose registers AX, BX, CX and DX (each split into high and low bytes like AH and AL). SI and DI are index registers, SP and BP are for the stack, and CS, DS, SS and ES are segment registers.

**Flags.** The flags register stores the result status of the last operation: Zero Flag (ZF) when the result is zero, Carry Flag (CF) for unsigned overflow, Sign Flag (SF) for a negative result, Overflow Flag (OF) for signed overflow. Conditional jumps like `JE`, `JNE`, `JG` and `JC` read these flags.

**Addressing modes I need to remember:**

- Immediate: `MOV AX, 5`
- Register: `MOV AX, BX`
- Direct: `MOV AX, [1234h]`
- Register indirect: `MOV AX, [BX]`
- Based/indexed: `MOV AL, ARR[SI]`

**Loops.** `LOOP` decreases CX by one and jumps back while CX is not zero, so I set CX to the number of repetitions first.

Example: sum of an array in EMU8086:

```asm
.MODEL SMALL
.DATA
ARR DB 2, 4, 6, 8
SUM DB 0
.CODE
MAIN PROC
    MOV AX, @DATA
    MOV DS, AX
    MOV CX, 4
    MOV SI, 0
    MOV AL, 0
NEXT:
    ADD AL, ARR[SI]
    INC SI
    LOOP NEXT
    MOV SUM, AL
    MOV AH, 4CH
    INT 21H
MAIN ENDP
END MAIN
```

**Input/output macros.** Writing a macro for printing a string or reading a character saves repeating the same `INT 21H` setup again and again. Most of my errors in EMU8086 were syntax mistakes and macro errors, so I always check the macro definition first when something weird happens.

## 3. Parallel and Distributed Computing (PDC)

PDC is the subject I find the most theory-heavy, with a lot of concepts and architecture topics, so I keep these notes short.

- **Parallel computing:** many processors or cores work on parts of one problem at the same time, usually sharing memory, to finish faster.
- **Distributed computing:** many separate computers connected by a network cooperate on a task; each has its own memory and they communicate by sending messages.
- **Flynn's taxonomy:** SISD (single instruction, single data), SIMD (one instruction on many data items, like GPUs), MISD (rare), MIMD (many independent processors, like multicore CPUs and clusters).
- **Shared memory vs distributed memory:** shared memory is easier to program but harder to scale; distributed memory scales better but needs message passing.
- **Speedup and Amdahl's law:** the serial part of a program limits how much faster it can get, no matter how many processors you add.
- **Common problems:** race conditions, deadlocks, load imbalance and communication overhead.

My trick for PDC: link every concept to a real example (a GPU for SIMD, a cluster for distributed memory), because the pure definitions all start to sound the same.
