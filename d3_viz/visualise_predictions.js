

function visualise_prediction(selected_year){

    var width = 800,
        height = 800,
        padding_center = 100,
        radius = Math.min(width, height) - 2*padding_center - 10  //radius is at most min of two above values /2

        ;


    var x = d3.scaleLinear()
        .domain([1,365])
        .range([0, 2 * Math.PI]);  //maps [0,364] to [0, 2*pi]

    var y = d3.scaleLinear() // scaleSqrt
        .domain([0,24])
        .range([0, radius / 2 ]);


    var color = d3.scaleOrdinal(d3.schemeCategory20);

    var arc = d3.arc() // Partition is "rotated" due to assignment of x,y values to arc
        .startAngle(function(d) { return Math.PI + Math.max(0, Math.min(2 * Math.PI, x(d.day_year))); }) //day_year defines angle
        .endAngle(function(d) { return Math.PI +Math.max(0, Math.min(2 * Math.PI, x(d.day_year + 1))); })
        .innerRadius(function(d) { return Math.max(0,  padding_center + y(d.hour)); }) // hour defines height
        .outerRadius(function(d) { return Math.max(0, padding_center + y(24)); }); //not sure why y(hour+1) leads to weird results. To be checked


    var svg = d3.select("body").append("svg")
        .attr("width", width)
        .attr("height", height)
      .append("g")
        .attr("transform", "translate(" + width / 2 + "," + (height ) / 2 + ")") // re-center chart
        .attr("id", "main_g");

    d3.csv("visualise_predictions.csv", function(error, dataset) {
      if (error) throw error;

      var dataset_filter = dataset.filter(function(d){return d.year === selected_year})

      //console.log(dataset_filter[0]);

      var g = svg.selectAll("g")
       .data(dataset_filter)
       .enter().append("g")
       .attr("id", function(d) { return d.date ; })
       .attr("day", function(d) { return d.day ; })
       .attr("hour", function(d) { return d.hour ; })
       .attr("day_year", function(d) { return d.day_year ; })


      var path = g.append("path")
          .attr("d", arc)
          .style("fill", function(d) { return d.color; })
          //.on("click", toggle_data_selection)
        .append("title")
          .text(function(d) { return d.date ; });

    var text = d3.select("#main_g").append("text");
    text.append('tspan')
        .attr("text-anchor", "middle")
        .attr("font-family", 'verdana')
        .attr("font-size", 35)
        .text("Year " + selected_year);

    //var month_label = svg.selectAll('g[day = 1]')
    var month_label = svg.selectAll("g").filter(function(d){return d.day === "1" && d.hour === "23"} )
        .append("text")
        .attr("transform", function(d) {
        //d.outerRadius =  padding_center + y(24) +1000 ; // Set Outer Coordinate
        //d.innerRadius =  padding_center + y(d.hour)+ 50; // Set Inner Coordinate
        return "translate(" + arc.centroid(d) + ")";})
        .text("gna " )


    var monthlabels = [
    {Month: "Jan", day: 1 },
    {Month: "Feb", day: 32 }

    ];


    });



};

visualise_prediction("2015")