"""Company organizational structure example for Grafito graph database.

This example demonstrates:
- Hierarchical organizational structure
- Departments and teams
- Management and reporting relationships
- Skills and project assignments
- Complex queries on organizational data
"""

from grafito import GrafitoDatabase


def main():
    print("=== Company Organizational Structure Example ===\n")

    # Initialize database
    db = GrafitoDatabase(':memory:')

    # =========================================================================
    # Create Company and Departments
    # =========================================================================
    print("Building organizational structure...\n")

    techcorp = db.create_node(
        labels=['Company'],
        properties={
            'name': 'TechCorp',
            'founded': 2010,
            'industry': 'Technology',
            'size': 'large'
        }
    )

    # Departments
    engineering = db.create_node(
        labels=['Department'],
        properties={'name': 'Engineering', 'budget': 5000000}
    )

    sales = db.create_node(
        labels=['Department'],
        properties={'name': 'Sales', 'budget': 2000000}
    )

    hr = db.create_node(
        labels=['Department'],
        properties={'name': 'Human Resources', 'budget': 1000000}
    )

    # Connect departments to company
    db.create_relationship(engineering.id, techcorp.id, 'PART_OF')
    db.create_relationship(sales.id, techcorp.id, 'PART_OF')
    db.create_relationship(hr.id, techcorp.id, 'PART_OF')

    # =========================================================================
    # Create Employees
    # =========================================================================

    # CEO
    ceo = db.create_node(
        labels=['Person', 'Employee', 'Executive'],
        properties={
            'name': 'Sarah Johnson',
            'title': 'CEO',
            'email': 'sarah.johnson@techcorp.com',
            'hire_date': '2010-01-01',
            'salary': 500000
        }
    )
    db.create_relationship(ceo.id, techcorp.id, 'WORKS_AT', {'since': 2010})

    # Engineering Department
    eng_director = db.create_node(
        labels=['Person', 'Employee', 'Manager'],
        properties={
            'name': 'Michael Chen',
            'title': 'Engineering Director',
            'email': 'michael.chen@techcorp.com',
            'hire_date': '2012-03-15',
            'salary': 250000
        }
    )
    db.create_relationship(eng_director.id, engineering.id, 'WORKS_IN', {'since': 2012})
    db.create_relationship(eng_director.id, ceo.id, 'REPORTS_TO')
    db.create_relationship(eng_director.id, engineering.id, 'MANAGES')

    # Engineering Team Leads
    backend_lead = db.create_node(
        labels=['Person', 'Employee', 'TeamLead'],
        properties={
            'name': 'Alice Williams',
            'title': 'Backend Team Lead',
            'email': 'alice.williams@techcorp.com',
            'hire_date': '2015-06-01',
            'salary': 180000
        }
    )
    db.create_relationship(backend_lead.id, engineering.id, 'WORKS_IN', {'since': 2015})
    db.create_relationship(backend_lead.id, eng_director.id, 'REPORTS_TO')

    frontend_lead = db.create_node(
        labels=['Person', 'Employee', 'TeamLead'],
        properties={
            'name': 'Bob Martinez',
            'title': 'Frontend Team Lead',
            'email': 'bob.martinez@techcorp.com',
            'hire_date': '2016-02-10',
            'salary': 175000
        }
    )
    db.create_relationship(frontend_lead.id, engineering.id, 'WORKS_IN', {'since': 2016})
    db.create_relationship(frontend_lead.id, eng_director.id, 'REPORTS_TO')

    # Engineers
    engineer1 = db.create_node(
        labels=['Person', 'Employee', 'Engineer'],
        properties={
            'name': 'Carol Davis',
            'title': 'Senior Backend Engineer',
            'email': 'carol.davis@techcorp.com',
            'hire_date': '2018-09-01',
            'salary': 150000
        }
    )
    db.create_relationship(engineer1.id, engineering.id, 'WORKS_IN', {'since': 2018})
    db.create_relationship(engineer1.id, backend_lead.id, 'REPORTS_TO')

    engineer2 = db.create_node(
        labels=['Person', 'Employee', 'Engineer'],
        properties={
            'name': 'David Lee',
            'title': 'Frontend Engineer',
            'email': 'david.lee@techcorp.com',
            'hire_date': '2019-04-15',
            'salary': 140000
        }
    )
    db.create_relationship(engineer2.id, engineering.id, 'WORKS_IN', {'since': 2019})
    db.create_relationship(engineer2.id, frontend_lead.id, 'REPORTS_TO')

    engineer3 = db.create_node(
        labels=['Person', 'Employee', 'Engineer'],
        properties={
            'name': 'Emma Wilson',
            'title': 'Backend Engineer',
            'email': 'emma.wilson@techcorp.com',
            'hire_date': '2020-11-01',
            'salary': 135000
        }
    )
    db.create_relationship(engineer3.id, engineering.id, 'WORKS_IN', {'since': 2020})
    db.create_relationship(engineer3.id, backend_lead.id, 'REPORTS_TO')

    # Sales Department
    sales_director = db.create_node(
        labels=['Person', 'Employee', 'Manager'],
        properties={
            'name': 'Frank Thompson',
            'title': 'Sales Director',
            'email': 'frank.thompson@techcorp.com',
            'hire_date': '2013-01-20',
            'salary': 220000
        }
    )
    db.create_relationship(sales_director.id, sales.id, 'WORKS_IN', {'since': 2013})
    db.create_relationship(sales_director.id, ceo.id, 'REPORTS_TO')
    db.create_relationship(sales_director.id, sales.id, 'MANAGES')

    sales_rep1 = db.create_node(
        labels=['Person', 'Employee', 'SalesRep'],
        properties={
            'name': 'Grace Kim',
            'title': 'Senior Sales Representative',
            'email': 'grace.kim@techcorp.com',
            'hire_date': '2017-07-01',
            'salary': 120000
        }
    )
    db.create_relationship(sales_rep1.id, sales.id, 'WORKS_IN', {'since': 2017})
    db.create_relationship(sales_rep1.id, sales_director.id, 'REPORTS_TO')

    # HR Department
    hr_manager = db.create_node(
        labels=['Person', 'Employee', 'Manager'],
        properties={
            'name': 'Helen Brown',
            'title': 'HR Manager',
            'email': 'helen.brown@techcorp.com',
            'hire_date': '2014-05-15',
            'salary': 180000
        }
    )
    db.create_relationship(hr_manager.id, hr.id, 'WORKS_IN', {'since': 2014})
    db.create_relationship(hr_manager.id, ceo.id, 'REPORTS_TO')
    db.create_relationship(hr_manager.id, hr.id, 'MANAGES')

    # =========================================================================
    # Create Skills
    # =========================================================================

    python_skill = db.create_node(labels=['Skill'], properties={'name': 'Python'})
    javascript_skill = db.create_node(labels=['Skill'], properties={'name': 'JavaScript'})
    react_skill = db.create_node(labels=['Skill'], properties={'name': 'React'})
    sql_skill = db.create_node(labels=['Skill'], properties={'name': 'SQL'})
    leadership_skill = db.create_node(labels=['Skill'], properties={'name': 'Leadership'})

    # Assign skills to employees
    db.create_relationship(backend_lead.id, python_skill.id, 'HAS_SKILL', {'years': 10})
    db.create_relationship(backend_lead.id, sql_skill.id, 'HAS_SKILL', {'years': 8})
    db.create_relationship(backend_lead.id, leadership_skill.id, 'HAS_SKILL', {'years': 5})

    db.create_relationship(frontend_lead.id, javascript_skill.id, 'HAS_SKILL', {'years': 8})
    db.create_relationship(frontend_lead.id, react_skill.id, 'HAS_SKILL', {'years': 6})
    db.create_relationship(frontend_lead.id, leadership_skill.id, 'HAS_SKILL', {'years': 4})

    db.create_relationship(engineer1.id, python_skill.id, 'HAS_SKILL', {'years': 5})
    db.create_relationship(engineer1.id, sql_skill.id, 'HAS_SKILL', {'years': 4})

    db.create_relationship(engineer2.id, javascript_skill.id, 'HAS_SKILL', {'years': 4})
    db.create_relationship(engineer2.id, react_skill.id, 'HAS_SKILL', {'years': 3})

    db.create_relationship(engineer3.id, python_skill.id, 'HAS_SKILL', {'years': 3})

    # =========================================================================
    # Query 1: Show Organizational Hierarchy
    # =========================================================================
    print("=" * 60)
    print("Query 1: Organizational Hierarchy")
    print("=" * 60)

    print(f"\nCEO: {ceo.properties['name']}")

    # Find direct reports to CEO
    ceo_reports = db.get_neighbors(ceo.id, direction='incoming', rel_type='REPORTS_TO')
    for director in ceo_reports:
        print(f"  ├─ {director.properties['title']}: {director.properties['name']}")

        # Find their reports
        team_reports = db.get_neighbors(director.id, direction='incoming', rel_type='REPORTS_TO')
        for i, member in enumerate(team_reports):
            is_last = (i == len(team_reports) - 1)
            prefix = "  │  └─" if is_last else "  │  ├─"
            print(f"{prefix} {member.properties['title']}: {member.properties['name']}")

            # Find their reports
            sub_reports = db.get_neighbors(member.id, direction='incoming', rel_type='REPORTS_TO')
            if sub_reports:
                for j, subreport in enumerate(sub_reports):
                    is_sub_last = (j == len(sub_reports) - 1)
                    sub_prefix = "  │     └─" if is_last and is_sub_last else "  │     ├─"
                    print(f"{sub_prefix} {subreport.properties['title']}: {subreport.properties['name']}")

    # =========================================================================
    # Query 2: Find All Employees in Engineering
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 2: Engineering Department")
    print("=" * 60)

    eng_employees = db.get_neighbors(engineering.id, direction='incoming', rel_type='WORKS_IN')
    print(f"\n{len(eng_employees)} employees in Engineering:")
    for emp in eng_employees:
        title = emp.properties['title']
        name = emp.properties['name']
        salary = emp.properties['salary']
        print(f"  - {name} - {title} (${salary:,})")

    # =========================================================================
    # Query 3: Find Employees with Python Skills
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 3: Python Developers")
    print("=" * 60)

    python_rels = db.match_relationships(target_id=python_skill.id, rel_type='HAS_SKILL')
    print(f"\n{len(python_rels)} employees with Python skills:")
    for rel in python_rels:
        emp = db.get_node(rel.source_id)
        years = rel.properties.get('years', 0)
        print(f"  - {emp.properties['name']} ({years} years experience)")

    # =========================================================================
    # Query 4: Find Reporting Chain
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 4: Reporting Chain for Carol Davis")
    print("=" * 60)

    # Follow REPORTS_TO relationships up to CEO
    current = engineer1
    chain = [current]
    print(f"\nReporting chain:")
    print(f"  1. {current.properties['name']} ({current.properties['title']})")

    level = 2
    while True:
        managers = db.get_neighbors(current.id, direction='outgoing', rel_type='REPORTS_TO')
        if not managers:
            break
        current = managers[0]
        chain.append(current)
        print(f"  {level}. {current.properties['name']} ({current.properties['title']})")
        level += 1

    # =========================================================================
    # Query 5: Find Department Budget Total
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 5: Department Budgets and Headcount")
    print("=" * 60)

    departments = db.match_nodes(labels=['Department'])
    print(f"\nDepartments overview:")
    for dept in departments:
        dept_name = dept.properties['name']
        budget = dept.properties['budget']
        headcount = len(db.get_neighbors(dept.id, direction='incoming', rel_type='WORKS_IN'))
        print(f"\n  {dept_name}:")
        print(f"    Budget: ${budget:,}")
        print(f"    Headcount: {headcount}")
        if headcount > 0:
            avg_cost = budget / headcount
            print(f"    Avg budget per employee: ${avg_cost:,.0f}")

    # =========================================================================
    # Query 6: Find All Managers
    # =========================================================================
    print("\n" + "=" * 60)
    print("Query 6: All Managers and Their Teams")
    print("=" * 60)

    managers = db.match_nodes(labels=['Manager'])
    print(f"\n{len(managers)} managers in the organization:")
    for manager in managers:
        team = db.get_neighbors(manager.id, direction='incoming', rel_type='REPORTS_TO')
        dept_rels = db.match_relationships(source_id=manager.id, rel_type='MANAGES')
        dept_name = ""
        if dept_rels:
            dept = db.get_node(dept_rels[0].target_id)
            dept_name = f" ({dept.properties['name']})"

        print(f"\n  {manager.properties['name']}{dept_name}")
        print(f"    Team size: {len(team)}")
        if team:
            print(f"    Team members:")
            for member in team:
                print(f"      - {member.properties['name']} ({member.properties['title']})")

    # =========================================================================
    # Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("Company Statistics")
    print("=" * 60)

    total_employees = db.get_node_count(label='Employee')
    total_managers = db.get_node_count(label='Manager')
    total_departments = db.get_node_count(label='Department')
    total_skills = db.get_node_count(label='Skill')

    print(f"\nTotal Employees: {total_employees}")
    print(f"Total Managers: {total_managers}")
    print(f"Total Departments: {total_departments}")
    print(f"Total Skills Tracked: {total_skills}")

    # Calculate total payroll
    all_employees = db.match_nodes(labels=['Employee'])
    total_payroll = sum(emp.properties.get('salary', 0) for emp in all_employees)
    avg_salary = total_payroll / len(all_employees) if all_employees else 0

    print(f"\nTotal Payroll: ${total_payroll:,}")
    print(f"Average Salary: ${avg_salary:,.0f}")

    print(f"\nAll relationship types: {', '.join(db.get_all_relationship_types())}")

    # Cleanup
    db.close()
    print("\n" + "=" * 60)
    print("Company structure example completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
