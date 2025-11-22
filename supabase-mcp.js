#!/usr/bin/env node

/**
 * Supabase MCP Server
 * 
 * This server provides tools for interacting with Supabase through the Model Context Protocol.
 * It allows querying tables, running SQL, and managing Supabase resources.
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { createClient } from '@supabase/supabase-js';

// Load environment variables
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
  console.error('Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables are required');
  process.exit(1);
}

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

// Create MCP server
const server = new Server(
  {
    name: 'supabase-mcp-server',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List available tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: 'query_table',
        description: 'Query a Supabase table with optional filters, sorting, and pagination',
        inputSchema: {
          type: 'object',
          properties: {
            table: {
              type: 'string',
              description: 'Name of the table to query',
            },
            select: {
              type: 'string',
              description: 'Columns to select (default: *)',
              default: '*',
            },
            filters: {
              type: 'array',
              description: 'Array of filter conditions',
              items: {
                type: 'object',
                properties: {
                  column: { type: 'string' },
                  operator: { type: 'string', enum: ['eq', 'neq', 'gt', 'gte', 'lt', 'lte', 'like', 'ilike', 'is', 'in'] },
                  value: {},
                },
                required: ['column', 'operator', 'value'],
              },
            },
            order: {
              type: 'object',
              properties: {
                column: { type: 'string' },
                ascending: { type: 'boolean', default: true },
              },
            },
            limit: {
              type: 'number',
              description: 'Maximum number of rows to return',
            },
          },
          required: ['table'],
        },
      },
      {
        name: 'insert_rows',
        description: 'Insert one or more rows into a Supabase table',
        inputSchema: {
          type: 'object',
          properties: {
            table: {
              type: 'string',
              description: 'Name of the table',
            },
            rows: {
              type: 'array',
              description: 'Array of row objects to insert',
              items: {
                type: 'object',
              },
            },
          },
          required: ['table', 'rows'],
        },
      },
      {
        name: 'update_rows',
        description: 'Update rows in a Supabase table',
        inputSchema: {
          type: 'object',
          properties: {
            table: {
              type: 'string',
              description: 'Name of the table',
            },
            updates: {
              type: 'object',
              description: 'Object with columns to update',
            },
            filters: {
              type: 'array',
              description: 'Array of filter conditions to identify rows to update',
              items: {
                type: 'object',
                properties: {
                  column: { type: 'string' },
                  operator: { type: 'string' },
                  value: {},
                },
                required: ['column', 'operator', 'value'],
              },
            },
          },
          required: ['table', 'updates', 'filters'],
        },
      },
      {
        name: 'delete_rows',
        description: 'Delete rows from a Supabase table',
        inputSchema: {
          type: 'object',
          properties: {
            table: {
              type: 'string',
              description: 'Name of the table',
            },
            filters: {
              type: 'array',
              description: 'Array of filter conditions to identify rows to delete',
              items: {
                type: 'object',
                properties: {
                  column: { type: 'string' },
                  operator: { type: 'string' },
                  value: {},
                },
                required: ['column', 'operator', 'value'],
              },
            },
          },
          required: ['table', 'filters'],
        },
      },
      {
        name: 'execute_sql',
        description: 'Execute a raw SQL query (use with caution)',
        inputSchema: {
          type: 'object',
          properties: {
            query: {
              type: 'string',
              description: 'SQL query to execute',
            },
          },
          required: ['query'],
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  try {
    switch (name) {
      case 'query_table': {
        let query = supabase.from(args.table).select(args.select || '*');

        // Apply filters
        if (args.filters) {
          for (const filter of args.filters) {
            query = query[filter.operator](filter.column, filter.value);
          }
        }

        // Apply ordering
        if (args.order) {
          query = query.order(args.order.column, { ascending: args.order.ascending ?? true });
        }

        // Apply limit
        if (args.limit) {
          query = query.limit(args.limit);
        }

        const { data, error } = await query;

        if (error) throw error;

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(data, null, 2),
            },
          ],
        };
      }

      case 'insert_rows': {
        const { data, error } = await supabase
          .from(args.table)
          .insert(args.rows)
          .select();

        if (error) throw error;

        return {
          content: [
            {
              type: 'text',
              text: `Successfully inserted ${data.length} row(s):\n${JSON.stringify(data, null, 2)}`,
            },
          ],
        };
      }

      case 'update_rows': {
        let query = supabase.from(args.table).update(args.updates);

        // Apply filters
        for (const filter of args.filters) {
          query = query[filter.operator](filter.column, filter.value);
        }

        const { data, error } = await query.select();

        if (error) throw error;

        return {
          content: [
            {
              type: 'text',
              text: `Successfully updated ${data.length} row(s):\n${JSON.stringify(data, null, 2)}`,
            },
          ],
        };
      }

      case 'delete_rows': {
        let query = supabase.from(args.table).delete();

        // Apply filters
        for (const filter of args.filters) {
          query = query[filter.operator](filter.column, filter.value);
        }

        const { data, error } = await query.select();

        if (error) throw error;

        return {
          content: [
            {
              type: 'text',
              text: `Successfully deleted ${data.length} row(s):\n${JSON.stringify(data, null, 2)}`,
            },
          ],
        };
      }

      case 'execute_sql': {
        const { data, error } = await supabase.rpc('exec_sql', { sql: args.query });

        if (error) throw error;

        return {
          content: [
            {
              type: 'text',
              text: JSON.stringify(data, null, 2),
            },
          ],
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    return {
      content: [
        {
          type: 'text',
          text: `Error: ${error.message}`,
        },
      ],
      isError: true,
    };
  }
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error('Supabase MCP Server running on stdio');
}

main().catch((error) => {
  console.error('Server error:', error);
  process.exit(1);
});
